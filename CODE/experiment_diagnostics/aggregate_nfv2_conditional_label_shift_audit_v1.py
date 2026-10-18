"""Strict aggregation for the NFv2 conditional/label-shift audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "RESULTS" / "diagnostics" / "nfv2_conditional_label_shift_audit_v1"
PROTOCOL = "nfv2_conditional_label_shift_audit_v1"
DOMAINS = ("unsw", "cic", "bot", "ton")
REPRESENTATIONS = ("raw39", "relational_v1")
SEEDS = (2026, 2027, 2028)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(values):
    out = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def _mean(values):
    values = _finite(values)
    return float(np.mean(values)) if values else None


def _sd(values):
    values = _finite(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "rows": len(rows)}


def _load_domain_fingerprint(root: Path, domain_root: Path | None = None) -> dict[tuple[str, str, str], float]:
    """Load the fixed source-domain identity summary for relation reporting only."""
    path = (domain_root or root.parent / "domain_fingerprint") / "aggregate" / "domain_summary.csv"
    if not path.is_file():
        raise ValueError(f"required prior domain-fingerprint aggregate is missing: {path}")
    lookup = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") != "logreg":
                continue
            key = (row["outer_target"], row["representation"], row["task"])
            if key in lookup:
                raise ValueError(f"duplicate domain-fingerprint summary key: {key}")
            lookup[key] = float(row["macro_ovr_auroc_mean"])
    expected = {(t, r, task) for t in DOMAINS for r in REPRESENTATIONS for task in ("all", "benign", "attack")}
    if set(lookup) != expected:
        raise ValueError("domain-fingerprint aggregate lacks required logreg task coverage")
    return lookup


def collect(root: Path):
    coords = {}
    for p in root.glob("coordinates/target_*/*/seed_*.json"):
        d = _json(p)
        key = (d.get("outer_target"), d.get("representation"), int(d.get("seed")))
        if key in coords:
            raise ValueError(f"duplicate coordinate: {key}")
        coords[key] = d
    targets = {}
    for p in root.glob("targets/*.json"):
        d = _json(p); target = d.get("target")
        if target in targets: raise ValueError(f"duplicate target audit: {target}")
        targets[target] = d
    return coords, targets


def _check_coordinate(d: dict, key):
    target, rep, seed = key
    if d.get("schema_version") != 1 or d.get("protocol") != PROTOCOL:
        raise ValueError(f"{key}: schema/protocol mismatch")
    if (d.get("outer_target"), d.get("representation"), int(d.get("seed"))) != key:
        raise ValueError(f"{key}: coordinate identity mismatch")
    if d.get("target_labels_used_for_training_or_selection") is not False:
        raise ValueError(f"{key}: target labels entered selection")
    if d.get("target_labels_read_posthoc") is not True:
        raise ValueError(f"{key}: missing post-hoc label declaration")
    for section in ("metrics", "score_distributions", "mechanism_summary", "prior_shift"):
        if section not in d: raise ValueError(f"{key}: missing {section}")


def _check_target(d: dict, target):
    if d.get("schema_version") != 1 or d.get("protocol") != PROTOCOL or d.get("target") != target:
        raise ValueError(f"{target}: target audit schema/identity mismatch")
    if d.get("target_labels_used_for_training_or_selection") is not False:
        raise ValueError(f"{target}: target labels entered selection")
    if d.get("target_labels_read_posthoc") is not True:
        raise ValueError(f"{target}: missing post-hoc label declaration")
    if d.get("n_coordinates") != len(REPRESENTATIONS) * len(SEEDS):
        raise ValueError(f"{target}: incomplete target coordinates")


def aggregate(root: Path, domain_root: Path | None = None) -> dict:
    root = root.resolve()
    manifest = _json(root / "protocol_manifest.json")
    if manifest.get("protocol") != PROTOCOL or manifest.get("external_data_allowed") is not False:
        raise ValueError("protocol manifest is not frozen to four-domain post-hoc audit")
    coords, targets = collect(root)
    domain_fingerprint = _load_domain_fingerprint(root, domain_root)
    expected = {(t, r, s) for t in DOMAINS for r in REPRESENTATIONS for s in SEEDS}
    if set(coords) != expected:
        raise ValueError(json.dumps({"missing": sorted(expected - set(coords)),
                                     "unexpected": sorted(set(coords) - expected)}))
    if set(targets) != set(DOMAINS):
        raise ValueError(f"target audit coverage mismatch: {sorted(targets)}")
    for key, d in coords.items(): _check_coordinate(d, key)
    for target, d in targets.items(): _check_target(d, target)

    rows = []
    for key in sorted(coords):
        d = coords[key]
        m = d["metrics"]; s = d["score_distributions"]; o = d["score_orientation"]
        fs = d["mechanism_summary"]["conditional_feature_shift"]
        c = d["mechanism_summary"]["exact_feature_label_conflict"]
        fp = {task: domain_fingerprint[(key[0], key[1], task)] for task in ("all", "benign", "attack")}
        rows.append({
            "outer_target": key[0], "representation": key[1], "seed": key[2],
            "source_attack_rate": d["source_train"]["pooled"],
            "target_attack_rate": d["target_test"]["attack_rate"], "prior_shift": d["prior_shift"],
            "source_attack_score_mean": s["source_validation"]["attack"]["mean"],
            "source_benign_score_mean": s["source_validation"]["benign"]["mean"],
            "target_attack_score_mean": s["target_test"]["attack"]["mean"],
            "target_benign_score_mean": s["target_test"]["benign"]["mean"],
            "source_score_margin": o["source_attack_minus_benign_mean"],
            "target_score_margin": o["target_attack_minus_benign_mean"],
            "target_attack_below_benign_mean": int(o["target_attack_below_benign_mean"]),
            "row_auroc": m["row_auroc"], "group_auroc": m["group_auroc"],
            "row_auprc": m["row_auprc"], "group_auprc": m["group_auprc"],
            "row_partial_auc_1pct": m["row_partial_auc_1pct"],
            "group_partial_auc_1pct": m["group_partial_auc_1pct"],
            "row_recall_at_source_threshold": m["row_recall_at_source_threshold"],
            "group_recall_at_source_threshold": m["group_recall_at_source_threshold"],
            "row_false_alerts_per_million_benign": m["row_false_alerts_per_million_benign"],
            "group_false_alerts_per_million_benign": m["group_false_alerts_per_million_benign"],
            "benign_mean_abs_smd": fs["benign"]["mean_abs_smd"],
            "attack_mean_abs_smd": fs["attack"]["mean_abs_smd"],
            "benign_max_abs_smd": fs["benign"]["max_abs_smd"],
            "attack_max_abs_smd": fs["attack"]["max_abs_smd"],
            "exact_match_rate": c["raw_row"]["match_rate"],
            "exact_opposite_label_rate": c["raw_row"]["any_opposite_label_rate"],
            "exact_only_opposite_label_rate": c["raw_row"]["only_opposite_label_rate"],
            "group_exact_opposite_label_rate": c["duplicate_group_equal"]["any_opposite_label_rate"],
            "domain_all_logreg_auroc": fp["all"],
            "domain_benign_logreg_auroc": fp["benign"],
            "domain_attack_logreg_auroc": fp["attack"],
        })
    coord_csv = _write_csv(root / "aggregate" / "coordinate_rows.csv", rows)

    summary_rows = []
    groups = defaultdict(list)
    for row in rows: groups[(row["outer_target"], row["representation"])].append(row)
    for (target, rep), values in sorted(groups.items()):
        summary_rows.append({
            "outer_target": target, "representation": rep, "n_seeds": len(values),
            "row_auroc_mean": _mean([x["row_auroc"] for x in values]), "row_auroc_sd": _sd([x["row_auroc"] for x in values]),
            "group_auroc_mean": _mean([x["group_auroc"] for x in values]), "group_auroc_sd": _sd([x["group_auroc"] for x in values]),
            "row_auprc_mean": _mean([x["row_auprc"] for x in values]), "group_auprc_mean": _mean([x["group_auprc"] for x in values]),
            "target_score_margin_mean": _mean([x["target_score_margin"] for x in values]),
            "target_score_margin_sd": _sd([x["target_score_margin"] for x in values]),
            "prior_shift": _mean([x["prior_shift"] for x in values]),
            "benign_mean_abs_smd": _mean([x["benign_mean_abs_smd"] for x in values]),
            "attack_mean_abs_smd": _mean([x["attack_mean_abs_smd"] for x in values]),
            "row_recall_at_source_threshold_mean": _mean([x["row_recall_at_source_threshold"] for x in values]),
            "row_false_alerts_per_million_benign_mean": _mean([x["row_false_alerts_per_million_benign"] for x in values]),
            "row_inversion_count": int(sum(float(x["row_auroc"]) < 0.5 for x in values)),
            "group_inversion_count": int(sum(float(x["group_auroc"]) < 0.5 for x in values)),
            "exact_opposite_label_rate": _mean([x["exact_opposite_label_rate"] for x in values]),
            "domain_all_logreg_auroc": _mean([x["domain_all_logreg_auroc"] for x in values]),
            "domain_benign_logreg_auroc": _mean([x["domain_benign_logreg_auroc"] for x in values]),
            "domain_attack_logreg_auroc": _mean([x["domain_attack_logreg_auroc"] for x in values]),
        })
    summary_csv = _write_csv(root / "aggregate" / "summary_rows.csv", summary_rows)

    target_rows = []
    for target in DOMAINS:
        d = targets[target]
        target_rows.append({
            "target": target, "n_test_rows": d["fixed_test"]["n_rows"],
            "target_attack_rate": d["fixed_test"]["attack_rate"],
            "n_unique_groups": d["fixed_test"]["n_unique_groups"],
            "source_attack_rate": d["source_train"]["pooled"],
            "prior_shift": d["fixed_test"]["attack_rate"] - d["source_train"]["pooled"],
            "exact_match_rate": d["exact_feature_label_conflict"]["raw_row"]["match_rate"],
            "exact_opposite_label_rate": d["exact_feature_label_conflict"]["raw_row"]["any_opposite_label_rate"],
            "group_exact_opposite_label_rate": d["exact_feature_label_conflict"]["duplicate_group_equal"]["any_opposite_label_rate"],
            "benign_mean_abs_smd": d["conditional_feature_shift"]["benign"]["mean_abs_smd"],
            "attack_mean_abs_smd": d["conditional_feature_shift"]["attack"]["mean_abs_smd"],
            "benign_max_abs_smd": d["conditional_feature_shift"]["benign"]["max_abs_smd"],
            "attack_max_abs_smd": d["conditional_feature_shift"]["attack"]["max_abs_smd"],
        })
    target_csv = _write_csv(root / "aggregate" / "target_audit_rows.csv", target_rows)

    md = ["# NFv2 Conditional/Label-Shift Audit", "",
          "All rows are post-hoc summaries of frozen four-domain prediction artifacts; target labels were not used for training, selection, calibration, or stopping.", "",
          "## Frozen model coordinates", "",
          "| Target | Representation | Row AUROC | Group AUROC | Target score margin | Benign abs. SMD | Attack abs. SMD | Exact opposite-label rate | Domain AUROC (all/benign/attack) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary_rows:
        md.append(f"| {row['outer_target']} | {row['representation']} | {row['row_auroc_mean']:.4f} | {row['group_auroc_mean']:.4f} | {row['target_score_margin_mean']:.4f} | {row['benign_mean_abs_smd']:.4f} | {row['attack_mean_abs_smd']:.4f} | {row['exact_opposite_label_rate']:.4f} | {row['domain_all_logreg_auroc']:.4f}/{row['domain_benign_logreg_auroc']:.4f}/{row['domain_attack_logreg_auroc']:.4f} |")
    md.extend(["", "## Target-level mechanism summaries", "",
               "| Target | Attack prior | Prior shift | Exact match | Exact opposite label | Benign abs. SMD | Attack abs. SMD |",
               "|---|---:|---:|---:|---:|---:|---:|"])
    for row in target_rows:
        md.append(f"| {row['target']} | {row['target_attack_rate']:.4f} | {row['prior_shift']:.4f} | {row['exact_match_rate']:.4f} | {row['exact_opposite_label_rate']:.4f} | {row['benign_mean_abs_smd']:.4f} | {row['attack_mean_abs_smd']:.4f} |")
    md.extend(["", "Interpretation is descriptive: heterogeneous conditional shifts or exact conflicts can co-occur with inversion, but this audit does not identify a causal mechanism.", ""])
    md_path = root / "aggregate" / "summary.md"; md_path.parent.mkdir(parents=True, exist_ok=True); md_path.write_text("\n".join(md), encoding="utf-8")
    outputs = {"coordinate_rows": coord_csv, "summary_rows": summary_csv, "target_audit_rows": target_csv,
               "summary_md": {"path": str(md_path.resolve()), "sha256": _sha256(md_path)}}
    result = {"schema_version": 1, "protocol": PROTOCOL, "status": "ok",
              "n_coordinates": len(rows), "n_target_audits": len(target_rows),
              "external_data_used": False, "target_labels_used_for_training_or_selection": False,
              "target_labels_read_posthoc": True, "outputs": outputs}
    (root / "aggregate").mkdir(exist_ok=True)
    (root / "aggregate" / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--domain-root", type=Path, default=PROJECT_ROOT / 'RESULTS/reruns/domain_fingerprint')
    args = parser.parse_args()
    result = aggregate(Path(args.root), args.domain_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
