"""Recompute the paper's nine-seed Q/control attribution and seed sensitivity.

Reads the retained frozen nine-seed tables distributed in RESULTS_PAPER.
The numerical aggregation functions are retained from the original derivation;
this entry no longer requires the excluded historical twelve-seed sweep.
Use --dry-run for read-only arithmetic, or --output for JSON-only exports.
This recomputes saved metrics, not missing historical per-sample predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

Q_SWEEP = ROOT / "RESULTS_PAPER/RQ3_qtransport_unlabeled_repair/EXP06_qtransport_paired_source_lr/nine_seed/metrics_long.json"
CTL_SWEEP = ROOT / "RESULTS_PAPER/RQ3_qtransport_unlabeled_repair/EXP07_matched_native_control_attribution/nine_seed/metrics.json"
CAP_AUDIT = ROOT / "RESULTS_PAPER/RQ3_qtransport_unlabeled_repair/EXP06_qtransport_paired_source_lr/quantile_cap_audit/edges_cap_audit.json"
OUT = ROOT / "RESULTS/reruns/q_summary"

TARGETS = ("unsw", "cic", "bot", "ton")
DOMAIN_ORDER = {t: i for i, t in enumerate(TARGETS)}

# The reporting basis. Ordinary integers, no date-like values.
SEEDS = (1, 7, 42, 99, 1234, 2026, 2027, 2028, 55555)
# Retained because the Stage3 chapters report this triple and a reader comparing
# chapters needs the same three seeds. It is a subset of SEEDS, not a second basis.
TRIPLE = (2026, 2027, 2028)
EXCLUDED = ()  # Frozen inputs contain only the nine paper seeds.

# Machine keys. The reporting layer renames these; the data layer must not.
Q_ARM = "q_expert_median_inductive"
NATIVE_ARM = "native78_shared_expert_median"
ERM_ARM = "source_erm"
QCAPPED_ARM = "q117_expert_median"
PAIR_PROBE = ("source_erm", "native78_pooled")

METRIC = "group_auroc"
AGG_KEYS = ("row_auroc", "group_auroc", "row_auprc", "group_auprc",
            "row_tpr_at_fpr1", "group_tpr_at_fpr1")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    shutil.move(str(tmp), str(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    shutil.move(str(tmp), str(path))


def _mean_sd(v: np.ndarray) -> tuple[float, float]:
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) if v.size > 1 else 0.0)


def describe(v: np.ndarray) -> dict:
    m, sd = _mean_sd(v)
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {"n": int(v.size), "mean": m, "sd": sd, "min": float(v.min()),
            "q1": q1, "median": med, "q3": q3, "max": float(v.max()),
            "iqr": q3 - q1, "cv": (sd / abs(m)) if m else float("nan")}


def subset_reference(v: np.ndarray, k: int = 3) -> dict:
    """Exact distribution of the mean of every k-seed subset of the basis.

    Enumerating all C(9,3) = 84 subsets beats a bootstrap: the reference is exact,
    carries no rng of its own, and is precisely the quantity the paper would have
    reported had a different triple been drawn from this basis.
    """
    means = np.array([v[list(c)].mean()
                      for c in itertools.combinations(range(v.size), k)])
    m, sd = _mean_sd(means)
    return {"k": k, "n_subsets": int(means.size), "mean_of_means": m,
            "sd_of_subset_means": sd, "min": float(means.min()),
            "max": float(means.max()), "range": float(means.max() - means.min()),
            "p2.5": float(np.percentile(means, 2.5)),
            "p97.5": float(np.percentile(means, 97.5)), "_means": means}


def locate(ref_means: np.ndarray, value: float) -> dict:
    return {"value": value,
            "percentile": float((ref_means < value).mean() * 100.0),
            "n_below": int((ref_means < value).sum()),
            "n_above": int((ref_means > value).sum())}


def load_table() -> dict[tuple[str, str, int], dict]:
    table: dict[tuple[str, str, int], dict] = {}
    for r in json.loads(Q_SWEEP.read_text(encoding="utf-8")):
        table[(r["target"], r["method"], int(r["seed"]))] = r
    for r in json.loads(CTL_SWEEP.read_text(encoding="utf-8")):
        table[(r["target"], r["method"], int(r["source_seed"]))] = r
    return table


def series(table, target, method, seeds, metric=METRIC):
    out, missing = [], []
    for s in seeds:
        row = table.get((target, method, s))
        if row is None:
            missing.append(s)
            continue
        v = row.get(metric)
        if v is None or not math.isfinite(float(v)):
            missing.append(s)
            continue
        out.append(float(v))
    if missing:
        raise SystemExit(f"{target}/{method}: no value at seeds {missing}")
    return np.asarray(out, dtype=float)


def paired_delta(table, target, seeds, metric=METRIC) -> dict:
    q = series(table, target, Q_ARM, seeds, metric)
    out = {}
    for name, other_arm in (("q_minus_erm", ERM_ARM),
                            ("q_minus_native", NATIVE_ARM),
                            ("q_minus_qcapped_control", QCAPPED_ARM)):
        d = q - series(table, target, other_arm, seeds, metric)
        m, sd = _mean_sd(d)
        out[name] = {"compared_arms": [Q_ARM, other_arm], "mean": m, "sd": sd,
                     "min": float(d.min()), "max": float(d.max()),
                     "n_positive": int((d > 0).sum()),
                     "n_negative": int((d < 0).sum()), "n": int(d.size),
                     "sign_stable": bool((d > 0).all() or (d < 0).all()),
                     "per_seed": {str(s): float(x) for s, x in zip(seeds, d)}}
    return out


def check_pairing(table, seeds) -> dict:
    """Prove a seed selects the same source subsample in both runners."""
    per_target = {}
    for t in TARGETS:
        a = series(table, t, PAIR_PROBE[0], seeds)
        b = series(table, t, PAIR_PROBE[1], seeds)
        d = np.abs(a - b)
        per_target[t] = {"max_abs_discrepancy": float(d.max()),
                         "per_seed": {str(s): float(x) for s, x in zip(seeds, d)}}
        if d.max() > 1e-6:
            raise SystemExit(
                f"PAIRING BROKEN on {t}: {PAIR_PROBE[0]} and {PAIR_PROBE[1]} differ by "
                f"{d.max():.3e}; the two runners did not draw the same source subsample "
                f"for the same seed, so no paired delta is interpretable")
    return {"compared": list(PAIR_PROBE), "tolerance": 1e-6,
            "seeds_checked": list(seeds), "max_abs_discrepancy": per_target,
            "conclusion": (f"identical to numerical precision on all {len(TARGETS)} "
                           f"targets and all {len(seeds)} retained seeds: a seed fixes "
                           "the same source subsample in both runners, so the per-seed "
                           "paired deltas are valid")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-base RQ3 aggregates on nine seeds")
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the headline, write nothing")
    args = ap.parse_args()

    for p in (Q_SWEEP, CTL_SWEEP):
        if not p.is_file():
            raise SystemExit(f"missing sweep output: {p}")

    q_rows_all = json.loads(Q_SWEEP.read_text(encoding="utf-8"))
    c_rows_all = json.loads(CTL_SWEEP.read_text(encoding="utf-8"))
    keep = set(SEEDS)
    q_rows = [r for r in q_rows_all if int(r["seed"]) in keep]
    c_rows = [r for r in c_rows_all if int(r["source_seed"]) in keep]
    q_rows.sort(key=lambda r: (r["seed"], DOMAIN_ORDER[r["target"]], r["method"]))
    c_rows.sort(key=lambda r: (r["source_seed"], DOMAIN_ORDER[r["target"]], r["method"]))

    found = sorted({int(r["seed"]) for r in q_rows_all})
    if {int(r["source_seed"]) for r in c_rows_all} != keep:
        raise SystemExit("control inputs must contain exactly the nine paper seeds")
    for rows, seed_key in ((q_rows_all, "seed"), (c_rows_all, "source_seed")):
        keys = [(r["target"], int(r[seed_key]), r["method"]) for r in rows]
        if len(keys) != len(set(keys)) or {r["target"] for r in rows} != set(TARGETS):
            raise SystemExit("duplicate coordinates or unexpected target set")
    if set(found) != keep | set(EXCLUDED):
        raise SystemExit(f"unexpected seed set in the Q sweep: {found}")
    if len(q_rows) != len(SEEDS) * len(TARGETS) * len({r["method"] for r in q_rows}):
        raise SystemExit("retained Q rows are not a complete 9 x 4 x arms grid")
    if len(c_rows) != len(SEEDS) * len(TARGETS) * len({r["method"] for r in c_rows}):
        raise SystemExit("retained control rows are not a complete 9 x 4 x arms grid")

    table = load_table()
    pairing = check_pairing(table, SEEDS)

    # ---- per (target, arm) aggregation, same schema as the runner's own ------
    agg = []
    for target in TARGETS:
        for method in sorted({r["method"] for r in q_rows}):
            vals = [r for r in q_rows if r["target"] == target and r["method"] == method]
            if not vals:
                continue
            rec = {"target": target, "method": method, "n_seeds": len(vals),
                   "seeds": [int(r["seed"]) for r in vals]}
            for key in AGG_KEYS:
                if key not in vals[0]:
                    continue
                v = np.asarray([x[key] for x in vals], dtype=np.float64)
                rec[f"{key}_mean"] = float(v.mean())
                rec[f"{key}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                rec[f"{key}_per_seed"] = [float(x) for x in v]
            agg.append(rec)

    agg_lines = ["# Q-TRANSPORT v2 on the nine-seed reporting basis (exact quantile "
                 "edges, seeds " + ",".join(str(s) for s in SEEDS) + ")", "",
                 "Group AUROC mean +/- SD across seeds. `source_erm` is the paired "
                 "source-only LR control (report name: source-LR ERM; NOT the Stage3 "
                 "neural ERM checkpoint).", "",
                 "| target | method | group AUROC | row AUROC | group TPR@1% |",
                 "|---|---|---:|---:|---:|"]
    for r in agg:
        agg_lines.append(f"| {r['target']} | {r['method']} | "
                         f"{r['group_auroc_mean']:.4f} +/- {r['group_auroc_sd']:.4f} | "
                         f"{r['row_auroc_mean']:.4f} +/- {r['row_auroc_sd']:.4f} | "
                         f"{r['group_tpr_at_fpr1_mean']:.4f} |")

    # ---- attribution: paired deltas and the direction verdict ---------------
    arms_all = sorted({r["method"] for r in q_rows} | {r["method"] for r in c_rows})
    attribution: dict = {"metric": METRIC, "seeds": list(SEEDS), "targets": {},
                         "verdict": {}}
    att_lines = [f"# RQ3 paired attribution on nine seeds ({METRIC})", "",
                 f"Seeds: {list(SEEDS)}", "",
                 "A given seed draws the *same* source subsample in both runners",
                 "(`default_rng(seed + DOMAINS.index(target))`, 25000 rows per source",
                 "domain), so every delta below is paired per seed, not per mean.", "",
                 "Report names: `source_erm` = source-LR ERM, `native78_shared_expert_median`",
                 "= matched native per-source expert median, `q117_expert_median` = the",
                 "capped-quantile control arm used to bound the quantile-edge change.", ""]
    for t in TARGETS:
        att = paired_delta(table, t, SEEDS)
        attribution["targets"][t] = {
            "arm_means": {a: describe(series(table, t, a, SEEDS))
                          for a in (Q_ARM, NATIVE_ARM, ERM_ARM, QCAPPED_ARM)},
            "deltas": att}
        att_lines += [f"## target = {t}", "",
                      "| arm | mean | sd | min | max |",
                      "|---|---:|---:|---:|---:|"]
        for a, d in attribution["targets"][t]["arm_means"].items():
            att_lines.append(f"| {a} | {d['mean']:.4f} | {d['sd']:.4f} | "
                             f"{d['min']:.4f} | {d['max']:.4f} |")
        att_lines += ["", "| comparison | mean | sd | min | max | n positive | sign stable |",
                      "|---|---:|---:|---:|---:|---:|---|"]
        for name, d in att.items():
            att_lines.append(f"| {name} | {d['mean']:+.4f} | {d['sd']:.4f} | "
                             f"{d['min']:+.4f} | {d['max']:+.4f} | "
                             f"{d['n_positive']}/{d['n']} | "
                             f"{'yes' if d['sign_stable'] else 'NO'} |")
        att_lines.append("")

    stable_erm = {t: attribution["targets"][t]["deltas"]["q_minus_erm"]["sign_stable"]
                  for t in TARGETS}
    stable_nat = {t: attribution["targets"][t]["deltas"]["q_minus_native"]["sign_stable"]
                  for t in TARGETS}
    attribution["verdict"] = {
        "cross_runner_pairing_check": pairing,
        "q_minus_erm_sign_stable_per_target": stable_erm,
        "q_minus_erm_direction_stable_targets": sum(stable_erm.values()),
        "q_minus_native_sign_stable_per_target": stable_nat,
        "q_minus_native_direction_stable_targets": sum(stable_nat.values()),
        "q_minus_native_seed_counts": {
            t: attribution["targets"][t]["deltas"]["q_minus_native"]["n_positive"]
            for t in TARGETS},
        "rule": ("a direction may be asserted only for a target whose paired delta is "
                 "sign-stable over all nine seeds; targets whose mean is positive but "
                 "whose sign is not stable must be reported as within seed noise"),
    }
    att_lines += ["## Direction verdict", "",
                  f"- Q vs source-LR ERM: sign stable on "
                  f"{sum(stable_erm.values())}/{len(TARGETS)} targets "
                  f"({', '.join(t for t in TARGETS if stable_erm[t])})",
                  f"- Q vs matched native: sign stable on "
                  f"{sum(stable_nat.values())}/{len(TARGETS)} targets "
                  f"({', '.join(t for t in TARGETS if stable_nat[t])})", "",
                  "| target | Q-native mean | positive seeds | assertable direction |",
                  "|---|---:|---:|---|"]
    for t in TARGETS:
        d = attribution["targets"][t]["deltas"]["q_minus_native"]
        att_lines.append(f"| {t} | {d['mean']:+.4f} | {d['n_positive']}/{d['n']} | "
                         f"{'yes' if d['sign_stable'] else 'NO'} |")
    att_lines += ["", "## Cross-runner pairing integrity", "",
                  f"`{PAIR_PROBE[0]}` and `{PAIR_PROBE[1]}` are the same computation,",
                  "so agreement proves a seed selects the same source subsample in both",
                  "programs. Tolerance 1e-6.", "",
                  "| target | max abs discrepancy |", "|---|---:|"]
    for t in TARGETS:
        att_lines.append(f"| {t} | "
                         f"{pairing['max_abs_discrepancy'][t]['max_abs_discrepancy']:.2e} |")

    # ---- seed sensitivity over the nine-seed basis --------------------------
    sens: dict = {"metric": METRIC, "seeds": list(SEEDS), "arms": arms_all,
                  "nominal_triple": list(TRIPLE), "targets": {},
                  "subset_reference": {"method": "exact enumeration of all C(9,3)=84 "
                                                 "three-seed subsets of the basis",
                                       "k": 3, "n_subsets": 84,
                                       "bootstrap": False, "rng_used": False,
                                       "why": ("an exact reference has no sampling error "
                                               "of its own, which matters when the "
                                               "question is whether a three-seed mean "
                                               "is unusual")},
                  "verdict": {}}
    sens_lines = [f"# Nine-seed sensitivity analysis ({METRIC})", "",
                  f"Seeds: {list(SEEDS)}", "",
                  f"Nominal triple = {list(TRIPLE)}, a subset of the basis, retained so",
                  "that RQ3 can be read against the Stage3 chapters that report the same",
                  "three seeds. It is not a second reporting basis.", "",
                  "Reference distribution: the exact mean of every C(9,3) = 84 three-seed",
                  "subset of the basis. No bootstrap, no rng.", ""]
    spans = {}
    for t in TARGETS:
        tblock = {}
        sens_lines += [f"## target = {t}", "",
                       "| arm | n | mean | sd | min | q1 | median | q3 | max |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for a in arms_all:
            if not any(k[0] == t and k[1] == a for k in table):
                continue
            v = series(table, t, a, SEEDS)
            d = describe(v)
            tblock[a] = d
            sens_lines.append(f"| {a} | {d['n']} | {d['mean']:.4f} | {d['sd']:.4f} | "
                              f"{d['min']:.4f} | {d['q1']:.4f} | {d['median']:.4f} | "
                              f"{d['q3']:.4f} | {d['max']:.4f} |")
        sens_lines.append("")

        refs = {}
        sens_lines += [f"### How wide is a three-seed report on this basis? ({t})", "",
                       "| arm | 9-seed mean | 9-seed sd | sd of the 84 subset means | "
                       "full subset range | triple mean | triple percentile |",
                       "|---|---:|---:|---:|---:|---:|---:|"]
        for a in arms_all:
            if a not in tblock:
                continue
            v = series(table, t, a, SEEDS)
            ref = subset_reference(v, 3)
            triple_mean = float(series(table, t, a, TRIPLE).mean())
            loc = locate(ref["_means"], triple_mean)
            refs[a] = {"nine_seed_mean": tblock[a]["mean"], "nine_seed_sd": tblock[a]["sd"],
                       "subset_mean_sd": ref["sd_of_subset_means"],
                       "subset_range": ref["range"], "subset_min": ref["min"],
                       "subset_max": ref["max"], "subset_p2.5": ref["p2.5"],
                       "subset_p97.5": ref["p97.5"],
                       "triple_mean": triple_mean,
                       "triple_percentile": loc["percentile"],
                       "triple_n_below": loc["n_below"],
                       "triple_n_above": loc["n_above"]}
            sens_lines.append(f"| {a} | {tblock[a]['mean']:.4f} | {tblock[a]['sd']:.4f} | "
                              f"{ref['sd_of_subset_means']:.4f} | {ref['range']:.4f} | "
                              f"{triple_mean:.4f} | {loc['percentile']:.1f} |")
        sens_lines.append("")
        tblock["_subset_reference"] = refs
        sens["targets"][t] = tblock
        for a in (NATIVE_ARM, Q_ARM, ERM_ARM):
            if a in refs:
                spans[f"{t}/{a}"] = refs[a]["subset_range"]

    worst = max(spans.items(), key=lambda kv: kv[1])
    sens["verdict"] = {
        "three_seed_subset_range_per_key_arm": spans,
        "widest": {"arm": worst[0], "range": worst[1]},
        "sd_ratio_native_over_q": {
            t: (sens["targets"][t][NATIVE_ARM]["sd"] / sens["targets"][t][Q_ARM]["sd"]
                if sens["targets"][t][Q_ARM]["sd"] else float("nan"))
            for t in TARGETS},
        "reading": ("a three-seed mean drawn from this basis can differ from another "
                    "three-seed mean by up to the subset range shown above, which on the "
                    "matched native control arm is several times the nine-seed sd of the "
                    "Q arm; three seeds therefore cannot support a direction claim on "
                    "that arm, and the paper reports per-seed sign counts instead"),
    }
    sens_lines += ["## Verdict", "",
                   "Range of the 84 possible three-seed subset means, per key arm:", ""]
    for k, v in sorted(spans.items(), key=lambda kv: -kv[1]):
        sens_lines.append(f"- {k}: {v:.4f}")
    sens_lines += ["", "Nine-seed sd of the matched native control arm divided by the",
                   "nine-seed sd of the Q arm (how much more seed-sensitive the control is):",
                   ""]
    for t in TARGETS:
        sens_lines.append(f"- {t}: {sens['verdict']['sd_ratio_native_over_q'][t]:.2f}x")

    # ---- nominal-triple subset, for cross-chapter comparison ----------------
    triple_rows = ([r for r in q_rows if int(r["seed"]) in set(TRIPLE)]
                   + [r for r in c_rows if int(r["source_seed"]) in set(TRIPLE)])

    headline = {}
    for t in TARGETS:
        d = attribution["targets"][t]
        headline[t] = {
            "q_mean": d["arm_means"][Q_ARM]["mean"], "q_sd": d["arm_means"][Q_ARM]["sd"],
            "native_mean": d["arm_means"][NATIVE_ARM]["mean"],
            "native_sd": d["arm_means"][NATIVE_ARM]["sd"],
            "erm_mean": d["arm_means"][ERM_ARM]["mean"],
            "erm_sd": d["arm_means"][ERM_ARM]["sd"],
            "q_minus_erm": d["deltas"]["q_minus_erm"]["mean"],
            "q_minus_erm_positive_seeds":
                f"{d['deltas']['q_minus_erm']['n_positive']}/9",
            "q_minus_native": d["deltas"]["q_minus_native"]["mean"],
            "q_minus_native_positive_seeds":
                f"{d['deltas']['q_minus_native']['n_positive']}/9",
            "q_minus_native_sign_stable": d["deltas"]["q_minus_native"]["sign_stable"],
            "cap_effect_mean": d["deltas"]["q_minus_qcapped_control"]["mean"],
        }

    manifest = {
        "protocol": "nfv2_rq3_retained_nine_seed_reaggregation",
        "metric": METRIC, "seed_basis": list(SEEDS), "n_seeds": len(SEEDS),
        "nominal_triple": list(TRIPLE),
        "recompute_only": "Reaggregated retained metrics; no training or historical prediction recovery.",
        "rows": {"q_retained": len(q_rows), "control_retained": len(c_rows),
                 "q_excluded": 0, "control_excluded": 0, "triple_subset": len(triple_rows)},
        "inputs": {"q": {"path": str(Q_SWEEP.relative_to(ROOT)), "sha256": _sha(Q_SWEEP)},
                   "control": {"path": str(CTL_SWEEP.relative_to(ROOT)), "sha256": _sha(CTL_SWEEP)}},
        "code_sha256": _sha(Path(__file__).resolve()),
        "reference_distribution": {"method": "exact enumeration", "k": 3, "n_subsets": 84},
        "headline": headline,
    }

    if args.dry_run:
        print(json.dumps({"headline": headline,
                          "rows": manifest["rows"],
                          "verdict": attribution["verdict"],
                          "subset_range": spans}, indent=2))
        print("\n(dry run; nothing written)")
        return

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "q_metrics_long.json", q_rows)
    _write_json(out / "q_aggregate.json", agg)
    _write_json(out / "control_metrics.json", c_rows)
    _write_json(out / "attribution.json", attribution)
    _write_json(out / "seed_sensitivity.json", sens)
    _write_json(out / "triple_subset.json", triple_rows)
    _write_json(out / "manifest.json", manifest)

    print("\n".join(att_lines[-14:]))
    print()
    print("\n".join(sens_lines[-12:]))
    print(f"\nwrote 7 JSON files into {out.relative_to(ROOT)}/")
    print(json.dumps({"q_rows": len(q_rows), "control_rows": len(c_rows),
                      "excluded_q_rows": len(q_rows_all) - len(q_rows),
                      "direction_stable_q_vs_erm": sum(stable_erm.values()),
                      "direction_stable_q_vs_native": sum(stable_nat.values())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
