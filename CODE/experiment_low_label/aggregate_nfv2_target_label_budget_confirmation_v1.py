"""Aggregate the complete NFv2 target-label budget experiment grid.

The adaptation summary supplies N=0, 50, 100, 500, 1000 coordinates.
This script combines them with the low-budget fixed-schedule coordinates
(N=10, 20), checks completeness, and emits cell summaries and gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V5_RUNS = PROJECT_ROOT / "RESULTS" / "reruns" / "adaptation_summary" / "runs.csv"
DEFAULT_LOW_ROOT = PROJECT_ROOT / "RESULTS" / "reruns" / "low_budget"
DEFAULT_OUTPUT = PROJECT_ROOT / "RESULTS/reruns/deployment_summary"
TARGETS = ("unsw", "cic", "bot", "ton")
MODES = ("full_ft", "target_only")
V5_BUDGETS = (0, 50, 100, 500, 1000)
LOW_BUDGETS = (10, 20)


def _mean_sd(values: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _v5_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = pd.read_csv(path)
    required = {"target", "n_labeled", "mode", "checkpoint_seed", "support_seed", "auroc", "auprc"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"v5 runs.csv missing columns: {missing}")
    source = source[source["n_labeled"].isin(V5_BUDGETS)].copy()
    source["result_source"] = "revision_fewshot_v5_aggregate"
    source["protocol"] = source.get("protocol_version", "group_fixed_test_adaptation_v5")
    source["adaptation_seed"] = source.get("adaptation_seed", source["checkpoint_seed"])
    source["auroc_row"] = source["auroc"]
    source["auroc_group"] = source.get("unique_auroc", np.nan)
    source["auprc_row"] = source["auprc"]
    source["auprc_group"] = source.get("unique_auprc", np.nan)
    source["source_threshold_fpr_row"] = source.get("source_threshold_target_fpr_1pct", np.nan)
    source["source_threshold_tpr_row"] = source.get("source_threshold_target_tpr_1pct", np.nan)
    source["source_threshold_precision_row"] = source.get("source_threshold_target_precision_1pct", np.nan)
    source["alerts_per_million_row"] = source.get("source_threshold_alerts_per_million_1pct", np.nan)
    source["false_alerts_per_million_benign_row"] = source["source_threshold_fpr_row"] * 1_000_000
    # The v5 worst-source threshold is a different ROW operating point. It
    # cannot be relabelled as duplicate-group evaluation. Preserve it under
    # explicit names; group operating metrics were not recorded in v5.
    source["source_worst_threshold_fpr_row"] = source.get("source_worst_threshold_target_fpr_1pct", np.nan)
    source["source_worst_threshold_tpr_row"] = source.get("source_worst_threshold_target_tpr_1pct", np.nan)
    source["source_worst_threshold_alerts_per_million_row"] = source.get("source_worst_threshold_alerts_per_million_1pct", np.nan)
    for col in ("source_threshold_fpr_group", "source_threshold_tpr_group",
                "alerts_per_million_group", "false_alerts_per_million_benign_group"):
        source[col] = np.nan
    source["target_oracle_tpr_row"] = source.get("target_oracle_tpr_at_fpr_1pct", np.nan)
    source["target_oracle_tpr_group"] = np.nan
    source["target_brier_row"] = np.nan
    source["target_ece_row"] = np.nan
    source["latency_ms_per_1000_rows"] = np.nan
    source["throughput_rows_per_sec"] = np.nan
    source["support_group_overlap"] = 0
    source["target_labels_used_for_adaptation"] = source["n_labeled"] > 0
    # v5 is the original target-supervised adaptation track: support labels
    # select the adaptation hyperparameters/early-stop point.  N=0 remains
    # zero-shot and the new N=10/20 track is source-scheduled below.
    source["target_labels_used_for_selection"] = source["n_labeled"] > 0
    source["external_data_used"] = False
    keep = _canonical_columns()
    return source.reindex(columns=keep)


def _nested_get(payload: dict, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _low_rows(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/n*/**/result.json")) if root.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "success" or payload.get("protocol") != "nfv2_target_label_budget_confirmation_v1":
            continue
        target = str(payload.get("target"))
        n = int(payload.get("n_labeled"))
        mode = str(payload.get("mode"))
        block = payload.get("target_fixed_test", {})
        dep = payload.get("deployment", {})
        row = _nested_get(dep, "fpr_0.01", "row", "target") or {}
        group = _nested_get(dep, "fpr_0.01", "group", "target") or {}
        cal = payload.get("calibration", {})
        support_audit = payload.get("support_audit", {})
        records.append({
            "result_type": "adapted",
            "path": str(path.resolve()),
            "protocol": payload.get("protocol"),
            "target": target,
            "n_labeled": n,
            "mode": mode,
            "checkpoint_seed": int(payload.get("checkpoint_seed")),
            "support_seed": int(payload.get("support_seed")),
            "adaptation_seed": int(payload.get("optimization_seed", payload.get("checkpoint_seed"))),
            "auroc_row": block.get("auroc"),
            "auroc_group": block.get("unique_auroc"),
            "auprc_row": block.get("auprc"),
            "auprc_group": block.get("unique_auprc"),
            "source_threshold_fpr_row": row.get("fpr"),
            "source_threshold_tpr_row": row.get("tpr"),
            "source_threshold_precision_row": row.get("precision"),
            "alerts_per_million_row": row.get("alerts_per_million_flows"),
            "false_alerts_per_million_benign_row": row.get("false_alerts_per_million_benign"),
            "source_threshold_fpr_group": group.get("fpr"),
            "source_threshold_tpr_group": group.get("tpr"),
            "alerts_per_million_group": group.get("alerts_per_million_flows"),
            "false_alerts_per_million_benign_group": group.get("false_alerts_per_million_benign"),
            "target_oracle_tpr_row": block.get("tpr_at_fpr_1pct"),
            "target_oracle_tpr_group": block.get("unique_tpr_at_fpr_1pct"),
            "target_brier_row": cal.get("target_brier"),
            "target_ece_row": cal.get("target_ece"),
            "latency_ms_per_1000_rows": _nested_get(payload, "inference", "latency_ms_per_1000_rows"),
            "throughput_rows_per_sec": _nested_get(payload, "inference", "throughput_rows_per_sec"),
            "support_group_overlap": _nested_get(support_audit, "support_eval_group_overlap"),
            "target_labels_used_for_adaptation": payload.get("target_labels_used_for_training_or_selection", True),
            "target_labels_used_for_selection": payload.get("target_labels_used_for_model_or_threshold_selection", True),
            "external_data_used": payload.get("external_data_used", True),
            "result_source": "target_label_budget_confirmation_v1_low",
        })
    return pd.DataFrame(records, columns=_canonical_columns())


def _canonical_columns() -> list[str]:
    return [
        "result_type", "path", "protocol", "target", "n_labeled", "mode",
        "checkpoint_seed", "support_seed", "adaptation_seed",
        "auroc_row", "auroc_group", "auprc_row", "auprc_group",
        "source_threshold_fpr_row", "source_threshold_tpr_row", "source_threshold_precision_row",
        "alerts_per_million_row", "false_alerts_per_million_benign_row",
        "source_threshold_fpr_group", "source_threshold_tpr_group",
        "alerts_per_million_group", "false_alerts_per_million_benign_group",
        "source_worst_threshold_fpr_row", "source_worst_threshold_tpr_row",
        "source_worst_threshold_alerts_per_million_row",
        "target_oracle_tpr_row", "target_oracle_tpr_group",
        "target_brier_row", "target_ece_row", "latency_ms_per_1000_rows",
        "throughput_rows_per_sec", "support_group_overlap",
        "target_labels_used_for_adaptation", "target_labels_used_for_selection",
        "external_data_used", "result_source",
    ]


def _cell_summary(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in _canonical_columns() if c not in {
        "result_type", "path", "protocol", "target", "mode", "n_labeled", "checkpoint_seed",
        "support_seed", "adaptation_seed", "result_source", "target_labels_used_for_adaptation",
        "target_labels_used_for_selection", "external_data_used",
    }]
    records = []
    for (target, n, mode), group in rows.groupby(["target", "n_labeled", "mode"], sort=True):
        rec: dict[str, Any] = {"target": target, "n_labeled": int(n), "mode": mode, "n_cells": int(len(group))}
        for col in metric_cols:
            mean, sd = _mean_sd(group[col])
            rec[f"{col}_mean"] = mean
            rec[f"{col}_sd"] = sd
        records.append(rec)
    return pd.DataFrame(records)


def _paired_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "n_labeled", "checkpoint_seed", "support_seed"]
    left = rows[rows["mode"] == "full_ft"].copy()
    right = rows[rows["mode"] == "target_only"].copy()
    merged = left.merge(right, on=keys, suffixes=("_full_ft", "_target_only"), how="inner")
    metrics = ["auroc_row", "auroc_group", "auprc_row", "auprc_group",
               "source_threshold_fpr_row", "source_threshold_tpr_row",
               "alerts_per_million_row", "target_brier_row", "target_ece_row"]
    out = merged[keys].copy()
    for metric in metrics:
        out[f"delta_{metric}"] = (
            pd.to_numeric(merged[f"{metric}_full_ft"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_target_only"], errors="coerce")
        )
    return out


def _gate_summary(summary: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol": "nfv2_target_label_budget_confirmation_v1",
        "rank_floor": 0.70,
        "deployment_rule": "mean target FPR <= 0.05 and mean target TPR >= 0.20 in at least 3 of 4 domains at source-selected 1% FPR threshold",
        "budgets": {},
    }
    for n in sorted(summary["n_labeled"].unique()):
        block: dict[str, Any] = {}
        for mode in (("zero_shot",) if n == 0 else MODES):
            cell = summary[(summary["n_labeled"] == n) & (summary["mode"] == mode)]
            rank_targets = []
            deploy_targets = []
            for target in TARGETS:
                row = cell[cell["target"] == target]
                if row.empty:
                    continue
                row = row.iloc[0]
                rank_targets.append({"target": target,
                                     "row_auroc": _safe_float(row.get("auroc_row_mean")),
                                     "group_auroc": _safe_float(row.get("auroc_group_mean"))})
                fpr = _safe_float(row.get("source_threshold_fpr_row_mean"))
                tpr = _safe_float(row.get("source_threshold_tpr_row_mean"))
                deploy_targets.append({"target": target, "fpr": fpr, "tpr": tpr,
                                       "passes": bool(np.isfinite(fpr) and np.isfinite(tpr) and fpr <= 0.05 and tpr >= 0.20)})
            block[mode] = {
                "n_rank_targets": len(rank_targets),
                "rank_targets": rank_targets,
                "all_targets_row_ge_floor": len(rank_targets) == len(TARGETS) and all(x["row_auroc"] >= 0.70 for x in rank_targets),
                "all_targets_group_ge_floor": len(rank_targets) == len(TARGETS) and all(x["group_auroc"] >= 0.70 for x in rank_targets),
                "deployment_targets": deploy_targets,
                "deployment_pass_count": int(sum(x["passes"] for x in deploy_targets)),
                "deployment_gate": int(sum(x["passes"] for x in deploy_targets)) >= 3,
            }
        result["budgets"][str(int(n))] = block
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-runs", type=Path, default=V5_RUNS)
    parser.add_argument("--low-root", type=Path, default=DEFAULT_LOW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    old = _v5_rows(args.v5_runs)
    low = _low_rows(args.low_root)
    rows = pd.concat([old, low], ignore_index=True)
    rows["n_labeled"] = pd.to_numeric(rows["n_labeled"], errors="raise").astype(int)
    rows = rows.sort_values(["n_labeled", "target", "mode", "checkpoint_seed", "support_seed"], kind="stable")
    duplicate_keys = ["target", "n_labeled", "mode", "checkpoint_seed", "support_seed"]
    dup = rows.duplicated(duplicate_keys, keep=False)
    if bool(dup.any()):
        raise ValueError(f"duplicate coordinate rows: {rows.loc[dup, duplicate_keys].to_dict('records')[:5]}")
    expected = {(t, n, m, seed, support) for t, n, m, seed, support in
                itertools.product(TARGETS, (10, 20, 50, 100, 500, 1000), MODES,
                                  (2026, 2027, 2028), (11, 23, 37, 53, 71))}
    expected |= {(t, 0, 'zero_shot', seed, -1) for t, seed in itertools.product(TARGETS, (2026, 2027, 2028))}
    actual = {(r.target, int(r.n_labeled), r.mode, int(r.checkpoint_seed),
               -1 if pd.isna(r.support_seed) else int(r.support_seed)) for r in rows.itertuples()}
    if len(rows) != 732 or actual != expected:
        raise ValueError(f'Incomplete deployment matrix: missing {len(expected-actual)}, unexpected {len(actual-expected)} coordinates')
    if (pd.to_numeric(rows["support_group_overlap"], errors="coerce").fillna(0) != 0).any():
        raise ValueError("support/evaluation duplicate-group overlap detected")
    low_rows = rows[rows["result_source"] == "target_label_budget_confirmation_v1_low"]
    if low_rows["target_labels_used_for_selection"].fillna(True).astype(bool).any():
        raise ValueError("low-budget target labels were used for model/threshold selection")
    if rows["external_data_used"].fillna(True).astype(bool).any():
        raise ValueError("external data marked as used")
    summary = _cell_summary(rows)
    deltas = _paired_deltas(rows)
    gates = _gate_summary(summary)
    rows.to_csv(args.output_root / "coordinate_rows.csv", index=False, float_format="%.10g")
    summary.to_csv(args.output_root / "summary_rows.csv", index=False, float_format="%.10g")
    deltas.to_csv(args.output_root / "paired_deltas.csv", index=False, float_format="%.10g")
    (args.output_root / "budget_gates.json").write_text(json.dumps(gates, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# NFv2 Target-label Budget Confirmation",
        "",
        f"Coordinate rows: {len(rows)}; cell summaries: {len(summary)}; paired deltas: {len(deltas)}.",
        "Existing N=0/50/100/500/1000 coordinates are reused read-only from `revision_fewshot_v5_aggregate`; N=10/20 are included only when their fixed-source-schedule result files are complete.",
        "N counts labelled duplicate groups with balanced positive/negative support. Support groups and all sibling rows are excluded from the fixed target internal-test evaluation.",
        "The new N=10/20 coordinates use source-frozen schedules and thresholds; the reused v5 N=50/100/500/1000 coordinates retain their original target-support validation selection. The four target domains are dependent evaluation conditions, not independent replicates.",
        "",
        "## Cell summaries",
        "",
        "| N | Target | Mode | Cells | Row AUROC | Group AUROC | Row AUPRC | Source-threshold FPR | Source-threshold TPR | Alerts/M |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["n_labeled", "target", "mode"]).iterrows():
        lines.append(
            f"| {int(row['n_labeled'])} | {row['target']} | {row['mode']} | {int(row['n_cells'])} | "
            f"{_safe_float(row['auroc_row_mean']):.4f} +/- {_safe_float(row['auroc_row_sd']):.4f} | "
            f"{_safe_float(row['auroc_group_mean']):.4f} +/- {_safe_float(row['auroc_group_sd']):.4f} | "
            f"{_safe_float(row['auprc_row_mean']):.4f} +/- {_safe_float(row['auprc_row_sd']):.4f} | "
            f"{_safe_float(row['source_threshold_fpr_row_mean']):.4f} | "
            f"{_safe_float(row['source_threshold_tpr_row_mean']):.4f} | "
            f"{_safe_float(row['alerts_per_million_row_mean']):.0f} |"
        )
    lines += ["", "## Gate interpretation", "", "The machine-readable `budget_gates.json` is authoritative. A gate is descriptive and does not treat the four targets as independent statistical replicates.", ""]
    for n, block in gates["budgets"].items():
        for mode, val in block.items():
            lines.append(f"- N={n}, {mode}: row-floor={val['all_targets_row_ge_floor']}, group-floor={val['all_targets_group_ge_floor']}, deployment-pass-targets={val['deployment_pass_count']}/4.")
    lines += ["", "## Provenance", "", f"v5 input: `{args.v5_runs.resolve()}` SHA-256 `{_sha256(args.v5_runs)}`.", f"Low-budget root: `{args.low_root.resolve()}`."]
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "nfv2_target_label_budget_confirmation_v1",
        "v5_input": str(args.v5_runs.resolve()),
        "v5_sha256": _sha256(args.v5_runs),
        "low_root": str(args.low_root.resolve()),
        "n_coordinate_rows": int(len(rows)),
        "n_summary_rows": int(len(summary)),
        "n_paired_rows": int(len(deltas)),
        "outputs": [p.name for p in sorted(args.output_root.iterdir()) if p.is_file()],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
