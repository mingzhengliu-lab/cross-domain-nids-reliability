"""Strict aggregation for the NFv2 domain-fingerprint/relational-v1 audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "RESULTS" / "diagnostics" / "nfv2_domain_fingerprint_relational_v1"
PROTOCOL_VERSION = "nfv2_domain_fingerprint_relational_v1"
DOMAINS = ("unsw", "cic", "bot", "ton")
REPRESENTATIONS = ("raw39", "relational_v1")
TASKS = ("all", "benign", "attack")
DOMAIN_MODELS = ("logreg", "mlp")
SEEDS = (2026, 2027, 2028)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(values)) if values else float("nan")


def _sd(values):
    values = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _collect(root: Path):
    domain = {}
    for p in root.glob("domain/target_*/*/*/*/seed_*/result.json"):
        d = _json(p)
        key = (d["outer_target"], d["representation"], d["task"], d["model"], int(d["seed"]))
        if key in domain:
            raise ValueError(f"duplicate domain coordinate: {key}")
        domain[key] = d
    model = {}
    for p in root.glob("model/target_*/*/seed_*/result.json"):
        d = _json(p)
        key = (d["outer_target"], d["representation"], int(d["seed"]))
        if key in model:
            raise ValueError(f"duplicate model coordinate: {key}")
        model[key] = d
    return domain, model


def _check_manifest(root: Path):
    manifest = _json(root / "protocol_manifest.json")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol manifest version mismatch")
    if manifest.get("external_data_allowed") is not False:
        raise ValueError("external data is not frozen off")
    if not manifest.get("code_sha256") or not manifest.get("config_sha256"):
        raise ValueError("protocol manifest lacks code/config fingerprints")
    return manifest


def _check_result(result: dict, manifest: dict, kind: str):
    if result.get("status") != "success" or result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("result status or protocol version mismatch")
    if result.get("code_sha256") != manifest["code_sha256"]:
        raise ValueError("result code fingerprint mismatch")
    if result.get("config_sha256") != manifest["config_sha256"]:
        raise ValueError("result config fingerprint mismatch")
    if result.get("target_labels_used_for_training_or_selection") is not False:
        raise ValueError("target labels entered training or selection")
    if kind == "domain" and result.get("target_labels_read") is not False:
        raise ValueError("domain result read target labels")
    if kind == "model" and result.get("target_labels_read_after_model_and_threshold_freeze") is not True:
        raise ValueError("model result lacks post-freeze target-label audit")


def aggregate(root: Path) -> dict:
    manifest = _check_manifest(root)
    domain, model = _collect(root)
    for result in domain.values():
        _check_result(result, manifest, "domain")
    for result in model.values():
        _check_result(result, manifest, "model")
    expected_domain = {(o, r, t, m, s) for o in DOMAINS for r in REPRESENTATIONS for t in TASKS for m in DOMAIN_MODELS for s in SEEDS}
    expected_model = {(o, r, s) for o in DOMAINS for r in REPRESENTATIONS for s in SEEDS}
    missing_domain = sorted(expected_domain - set(domain))
    missing_model = sorted(expected_model - set(model))
    unexpected_domain = sorted(set(domain) - expected_domain)
    unexpected_model = sorted(set(model) - expected_model)
    if missing_domain or missing_model or unexpected_domain or unexpected_model:
        raise ValueError(json.dumps({"missing_domain": missing_domain, "missing_model": missing_model,
                                     "unexpected_domain": unexpected_domain, "unexpected_model": unexpected_model}, indent=2))

    domain_rows = []
    for (outer, rep, task, method, seed), result in sorted(domain.items()):
        metrics = result["source_validation_metrics"]
        domain_rows.append({
            "outer_target": outer, "representation": rep, "task": task, "model": method, "seed": seed,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_ovr_auroc": metrics["macro_ovr_auroc"],
            "log_loss": metrics["log_loss"],
            "permutation_baseline_auroc": result["permutation_importance"]["baseline_macro_ovr_auroc"],
            "target_posthoc_total": sum(result["target_posthoc_unlabeled_prediction_counts"]),
        })
    domain_summary = []
    groups = defaultdict(list)
    for row in domain_rows:
        groups[(row["outer_target"], row["representation"], row["task"], row["model"])].append(row)
    for key, rows in sorted(groups.items()):
        domain_summary.append({
            "outer_target": key[0], "representation": key[1], "task": key[2], "model": key[3],
            "n_seeds": len(rows),
            "balanced_accuracy_mean": _mean([x["balanced_accuracy"] for x in rows]),
            "balanced_accuracy_sd": _sd([x["balanced_accuracy"] for x in rows]),
            "macro_f1_mean": _mean([x["macro_f1"] for x in rows]),
            "macro_ovr_auroc_mean": _mean([x["macro_ovr_auroc"] for x in rows]),
            "macro_ovr_auroc_sd": _sd([x["macro_ovr_auroc"] for x in rows]),
            "log_loss_mean": _mean([x["log_loss"] for x in rows]),
        })

    model_rows = []
    for (outer, rep, seed), result in sorted(model.items()):
        metrics = result["metrics"]
        model_rows.append({
            "outer_target": outer, "representation": rep, "seed": seed,
            "row_auroc": metrics["auroc"], "group_auroc": metrics.get("unique_auroc", metrics["auroc"]),
            "row_auprc": metrics["auprc"], "group_auprc": metrics.get("unique_auprc", metrics["auprc"]),
            "row_partial_auc_1pct": metrics.get("partial_auc_1pct", float("nan")),
            "group_partial_auc_1pct": metrics.get("unique_partial_auc_1pct", float("nan")),
            "row_recall_at_source_1pct": metrics.get("source_threshold_recall_row", float("nan")),
            "group_recall_at_source_1pct": metrics.get("source_threshold_recall_group", float("nan")),
            "row_false_alerts_m": metrics.get("source_threshold_false_alerts_m_row", float("nan")),
            "group_false_alerts_m": metrics.get("source_threshold_false_alerts_m_group", float("nan")),
            "row_brier": metrics.get("brier_row", float("nan")), "group_brier": metrics.get("brier_group", float("nan")),
            "row_ece": metrics.get("ece_row", float("nan")), "utility": metrics.get("utility", float("nan")),
            "latency_ms_per_1000_rows": result.get("inference", {}).get("latency_ms_per_1000_rows", float("nan")),
            "throughput_rows_per_sec": result.get("inference", {}).get("throughput_rows_per_sec", float("nan")),
            "row_inverted": int(metrics["auroc"] < 0.5), "group_inverted": int(metrics.get("unique_auroc", metrics["auroc"]) < 0.5),
            "elapsed_sec": result["elapsed_sec"],
        })
    model_summary = []
    groups = defaultdict(list)
    for row in model_rows:
        groups[(row["outer_target"], row["representation"])].append(row)
    for key, rows in sorted(groups.items()):
        model_summary.append({
            "outer_target": key[0], "representation": key[1], "n_seeds": len(rows),
            "row_auroc_mean": _mean([x["row_auroc"] for x in rows]), "row_auroc_sd": _sd([x["row_auroc"] for x in rows]),
            "group_auroc_mean": _mean([x["group_auroc"] for x in rows]), "group_auroc_sd": _sd([x["group_auroc"] for x in rows]),
            "row_auprc_mean": _mean([x["row_auprc"] for x in rows]), "group_auprc_mean": _mean([x["group_auprc"] for x in rows]),
            "row_partial_auc_1pct_mean": _mean([x["row_partial_auc_1pct"] for x in rows]),
            "group_partial_auc_1pct_mean": _mean([x["group_partial_auc_1pct"] for x in rows]),
            "row_recall_at_source_1pct_mean": _mean([x["row_recall_at_source_1pct"] for x in rows]),
            "group_recall_at_source_1pct_mean": _mean([x["group_recall_at_source_1pct"] for x in rows]),
            "row_false_alerts_m_mean": _mean([x["row_false_alerts_m"] for x in rows]),
            "group_false_alerts_m_mean": _mean([x["group_false_alerts_m"] for x in rows]),
            "row_brier_mean": _mean([x["row_brier"] for x in rows]), "group_brier_mean": _mean([x["group_brier"] for x in rows]),
            "row_ece_mean": _mean([x["row_ece"] for x in rows]), "utility_mean": _mean([x["utility"] for x in rows]),
            "latency_ms_per_1000_rows_mean": _mean([x["latency_ms_per_1000_rows"] for x in rows]),
            "throughput_rows_per_sec_mean": _mean([x["throughput_rows_per_sec"] for x in rows]),
            "row_inversion_count": int(sum(x["row_inverted"] for x in rows)),
            "group_inversion_count": int(sum(x["group_inverted"] for x in rows)),
            "elapsed_sec_mean": _mean([x["elapsed_sec"] for x in rows]),
        })

    relations = []
    by_domain = {(r["outer_target"], r["representation"], r["task"], r["model"]): r for r in domain_summary}
    for row in model_summary:
        for task in TASKS:
            key = (row["outer_target"], row["representation"], task, "logreg")
            domain_row = by_domain[key]
            relations.append({
                "outer_target": row["outer_target"], "representation": row["representation"], "domain_task": task,
                "domain_model": "logreg", "domain_auroc": domain_row["macro_ovr_auroc_mean"],
                "attack_row_auroc": row["row_auroc_mean"], "attack_group_auroc": row["group_auroc_mean"],
                "attack_utility": row["utility_mean"], "row_inversion_count": row["row_inversion_count"],
                "group_inversion_count": row["group_inversion_count"],
            })
    aggregate_payload = {
        "status": "success", "protocol_version": PROTOCOL_VERSION,
        "expected_domain_coordinates": len(expected_domain), "observed_domain_coordinates": len(domain),
        "expected_model_coordinates": len(expected_model), "observed_model_coordinates": len(model),
        "domain_rows": domain_rows, "domain_summary": domain_summary,
        "model_rows": model_rows, "model_summary": model_summary,
        "mechanism_relations": relations,
        "external_data_read": False,
        "target_labels_used_for_training_or_selection": False,
    }
    aggregate_dir = root / "aggregate"; aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "summary.json").write_text(json.dumps(aggregate_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, rows in (("domain_rows.csv", domain_rows), ("domain_summary.csv", domain_summary), ("model_rows.csv", model_rows), ("model_summary.csv", model_summary), ("mechanism_relations.csv", relations)):
        if not rows: continue
        with (aggregate_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    return aggregate_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = aggregate(args.root)
    print(json.dumps({"status": result["status"], "domain": result["observed_domain_coordinates"], "model": result["observed_model_coordinates"]}, indent=2))


if __name__ == "__main__":
    main()
