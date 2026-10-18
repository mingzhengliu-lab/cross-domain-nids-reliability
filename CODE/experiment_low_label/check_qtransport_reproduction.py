"""Rerun one paper coordinate, or compare a completed full Q/control matrix.

Outputs are new reproduction artifacts, not recovered historical files.
No Markdown files are written. The original frozen results are read-only.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from CODE.experiment_low_label import run_qtransport_fair_cpu_v2 as q
from CODE.experiment_low_label import run_matched_expert_control as control

FROZEN = ROOT / "RESULTS_PAPER/RQ3_target_information_recovery"
SEEDS = q.DEFAULT_SEEDS
METRICS = ("row_auroc", "group_auroc", "row_auprc", "group_auprc",
           "row_tpr_at_fpr1", "group_tpr_at_fpr1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=q.DOMAINS, default="unsw")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=2026)
    parser.add_argument("--compare-only", action="store_true",
                        help="compare existing rerun artifacts without training again")
    parser.add_argument("--matrix", action="store_true",
                        help="compare all four targets and nine seeds; requires --compare-only")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "RESULTS/reruns/qtransport_reproduction_check")
    parser.add_argument("--report", type=Path,
                        help="write the comparison separately from the rerun artifacts")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.is_relative_to(FROZEN.resolve()) or out.is_relative_to((ROOT / "DATA").resolve()):
        parser.error("output must be separate from DATA and the frozen paper results")
    if args.matrix:
        if not args.compare_only:
            parser.error("--matrix requires --compare-only; run qtransport and matched-control first")
        comparisons = []
        for target in q.DOMAINS:
            for seed in SEEDS:
                path = out / "comparisons" / f"{target}_seed{seed}.json"
                main(["--compare-only", "--target", target, "--seed", str(seed),
                      "--output", str(out), "--report", str(path)])
                comparisons.append(json.loads(path.read_text()))
        report = {
            "scope": "Complete four-target, nine-source-seed Q and matched-control matrix.",
            "artifact_role": "Newly regenerated files checked against frozen historical metrics and SHA-256 records.",
            "targets": list(q.DOMAINS), "source_seeds": list(SEEDS),
            "coordinates": len(comparisons), "method_records": 36 * (8 + 11),
            "metric_values_compared": sum(c["metric_values_compared"] for r in comparisons
                                           for c in r["comparisons"].values()),
            "environment": comparisons[0]["environment"],
            "strict_reproduction_passed": all(r["strict_reproduction_passed"] for r in comparisons),
            "prediction_archive_hashes_all_match_historical": all(
                c["prediction_file_sha256_matches_historical"] for r in comparisons
                for c in r["comparisons"].values()),
            "source_index_archive_hashes_all_match_historical": all(
                c["source_indices_file_sha256_matches_historical"] for r in comparisons
                for c in r["comparisons"].values()),
            "max_absolute_error_by_metric": {
                metric: max(c["max_absolute_error_by_metric"].get(metric, 0.0)
                            for r in comparisons for c in r["comparisons"].values())
                for metric in METRICS},
            "comparisons": comparisons,
        }
        report["interpretation"] = (
            "All regenerated prediction/source-index archives match their historical SHA-256 "
            "records and all compared metrics match."
            if report["strict_reproduction_passed"] else
            "Full historical reproduction failed; preserve the original acceptance criterion.")
        report_path = args.report or out / "matrix_comparison.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(report_path), "passed": report["strict_reproduction_passed"]}), flush=True)
        return 0 if report["strict_reproduction_passed"] else 1
    coordinate = f"{args.target}_seed{args.seed}"
    if not args.compare_only and any((out / name / coordinate).exists() for name in ("qtransport", "matched_control")):
        parser.error("coordinate already exists; choose a new output directory")
    inputs = [p for p in q.DATA.rglob("*") if p.is_file()]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    report = {
        "scope": "One target and source seed; not a full nine-seed reproduction.",
        "artifact_role": "New rerun outputs, separate from historical frozen outputs.",
        "operation": "compare_existing_rerun" if args.compare_only else "train_and_compare",
        "target": args.target, "seed": args.seed, "source_rows_per_domain": 25000,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__},
        "comparisons": {},
    }
    for name, runner, experiment, table_name, seed_key in (
        ("qtransport", q, "EXP06_qtransport_paired_source_lr", "metrics_long.json", "seed"),
        ("matched_control", control, "EXP07_matched_native_control_attribution", "metrics.json", "source_seed"),
    ):
        print(f"{'Comparing' if args.compare_only else 'Running'} {name}: {coordinate}", flush=True)
        rows = (json.loads((out / name / coordinate / "metrics.json").read_text())
                if args.compare_only else runner.run_cell(args.target, args.seed, 25000, out / name))
        base = FROZEN / experiment / "nine_seed"
        frozen_rows = json.loads((base / table_name).read_text())
        expected = {r["method"]: r for r in frozen_rows
                    if r["target"] == args.target and int(r[seed_key]) == args.seed}
        assert {r["method"] for r in rows} == set(expected)
        metrics = [key for key in METRICS
                   if all(key in r and key in expected[r["method"]] for r in rows)]
        assert set(METRICS[:4]).issubset(metrics)
        errors = {metric: max(abs(float(r[metric]) - float(expected[r["method"]][metric]))
                              for r in rows) for metric in metrics}
        old_freeze = json.loads((base / "per_cell" / coordinate / "prediction_freeze.json").read_text())
        new_freeze = json.loads((out / name / coordinate / "prediction_freeze.json").read_text())
        report["comparisons"][name] = {
            "arms": len(rows), "metric_values_compared": len(rows) * len(metrics),
            "compared_metrics": metrics,
            "group_auroc_by_arm": {
                r["method"]: {"historical": expected[r["method"]]["group_auroc"],
                              "rerun": r["group_auroc"],
                              "difference": r["group_auroc"] - expected[r["method"]]["group_auroc"]}
                for r in rows},
            "max_absolute_error_by_metric": errors,
            "all_metrics_within_1e_8": all(value <= 1e-8 for value in errors.values()),
            "source_indices_file_sha256_matches_historical":
                old_freeze["source_indices_sha256"] == new_freeze["source_indices_sha256"],
            "prediction_file_sha256_matches_historical":
                old_freeze["prediction_sha256"] == new_freeze["prediction_sha256"],
            "historical_prediction_sha256": old_freeze["prediction_sha256"],
            "rerun_prediction_sha256": new_freeze["prediction_sha256"],
            "historical_array_contents_available": False,
            "archive_hash_interpretation":
                ("Regenerated NPZ bytes match the historical prediction and source-index "
                 "SHA-256 fingerprints; original files were absent before this rerun."
                 if old_freeze["source_indices_sha256"] == new_freeze["source_indices_sha256"]
                 and old_freeze["prediction_sha256"] == new_freeze["prediction_sha256"] else
                 "NPZ archive hashes can differ because of ZIP metadata; historical "
                 "arrays were absent, so equality is unverified when fingerprints differ."),
        }
        # Compare array values to the declared deterministic sampling algorithm.
        # This is not a comparison against missing historical array contents.
        rng = np.random.default_rng(args.seed + q.DOMAINS.index(args.target))
        reconstructed = {}
        with np.load(out / name / coordinate / "source_indices.npz") as actual:
            for domain in q.DOMAINS:
                if domain == args.target:
                    continue
                train_idx = np.load(q.DATA / "splits" / f"{domain}_split.npz")["train_idx"]
                expected_idx = train_idx[rng.choice(len(train_idx), min(25000, len(train_idx)), replace=False)]
                reconstructed[domain] = bool(np.array_equal(actual[domain], expected_idx))
        report["comparisons"][name]["source_index_values_match_declared_sampling"] = reconstructed
        if argv is None:
            print(json.dumps(report["comparisons"][name], indent=2), flush=True)
    with np.load(out / "qtransport" / coordinate / "frozen_predictions.npz") as a, \
            np.load(out / "matched_control" / coordinate / "frozen_predictions.npz") as b:
        report["paired_test_indices_identical"] = bool(np.array_equal(a["test_idx"], b["test_idx"]))
        report["paired_test_groups_identical"] = bool(np.array_equal(a["test_gid"], b["test_gid"]))
        report["pooled_baseline_score_max_absolute_difference"] = float(
            np.max(np.abs(a["source_erm"] - b["native78_pooled"])))
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
             for p in q.DATA.rglob("*") if p.is_file()}
    assert before == after, "Processed DATA changed during the run"
    report["processed_DATA_unchanged_during_operation"] = True
    report["passed"] = (
        all(x["all_metrics_within_1e_8"] and x["source_indices_file_sha256_matches_historical"]
            and x["prediction_file_sha256_matches_historical"]
            for x in report["comparisons"].values())
        and report["paired_test_indices_identical"] and report["paired_test_groups_identical"]
        and report["pooled_baseline_score_max_absolute_difference"] <= 1e-8)
    report["strict_reproduction_passed"] = report["passed"]
    report["interpretation"] = (
        "Strict historical reproduction is not established. Preserve the failed "
        "1e-8 criterion; report metric differences separately rather than relaxing "
        "the acceptance tolerance after seeing the rerun."
        if not report["passed"] else "Strict historical comparison passed.")
    report_path = args.report or out / f"comparison_{coordinate}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
