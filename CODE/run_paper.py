#!/usr/bin/env python3
"""Run manuscript checks, presentation, data verification or experiments.

Audit and tables need only the compact RESULTS_PAPER inputs. Training needs
processed data and, for neural methods, requirements-experiments.txt. Outputs
default to RESULTS/reruns; frozen results are never overwritten by this entry
point. See README.md for the data, checkpoint and historical-cache requirements.
Pass --help after a command for its own options, or --print-command to inspect
the command without importing training dependencies.
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
Q_SEEDS = "1,7,42,99,1234,2026,2027,2028,55555"
RERUNS = ROOT / "RESULTS/reruns"
COMMANDS = {
    "matrix": ("CODE/run_matrix.py", []),
    "summarize": ("CODE/analysis/summarize_reruns.py", []),
    "collect-results": ("CODE/analysis/collect_reruns.py", []),
    "support-controls": ("CODE/experiment_sensitivity/run_support_controls.py", []),
    "target-calibration": ("CODE/experiment_sensitivity/run_target_calibration.py", []),
    "domain-summary": ("CODE/experiment_diagnostics/aggregate_nfv2_domain_fingerprint_relational_v1.py", ["--root", str(RERUNS / "domain_fingerprint")]),
    "label-summary": ("CODE/experiment_diagnostics/aggregate_nfv2_conditional_label_shift_audit_v1.py", ["--root", str(RERUNS / "label_conflict"), "--domain-root", str(RERUNS / "domain_fingerprint")]),
    "audit": ("CODE/analysis/audit_results.py", []),
    "check-manuscript": ("CODE/analysis/check_manuscript.py", []),
    "tables": ("CODE/analysis/derive_paper_tables.py", []),
    "figures": ("CODE/figures/plot_empirical_figures.py", []),
    "verify-data": ("CODE/analysis/verify_data.py", []),
    "prepare-data": ("CODE/prepare_data.py", []),
    "stage3": ("CODE/experiment_stage3/stage3_run.py", ["--output_root", str(RERUNS / "stage3")]),
    "stage3-summary": ("CODE/experiment_stage3/stage3_aggregate.py", ["--stage3_dir", str(RERUNS / "stage3"), "--output_dir", str(RERUNS / "stage3_summary")]),
    "tabular": ("CODE/experiment_diagnostics/run_tabular_baselines.py", ["--out_dir", str(RERUNS / "tabular_baselines"), "--aggregate_dir", str(RERUNS / "tabular_aggregate")]),
    "anomaly": ("CODE/experiment_diagnostics/run_anomaly_benchmark.py", ["--output_root", str(RERUNS / "anomaly")]),
    "domain-fingerprint": ("CODE/experiment_diagnostics/run_nfv2_domain_fingerprint_relational_v1.py", ["--output-root", str(RERUNS / "domain_fingerprint")]),
    "label-conflict": ("CODE/experiment_diagnostics/run_nfv2_conditional_label_shift_audit_v1.py", ["--output_dir", str(RERUNS / "label_conflict"), "--model_root", str(RERUNS / "domain_fingerprint")]),
    "qtransport": ("CODE/experiment_low_label/run_qtransport_fair_cpu_v2.py", ["--seeds", Q_SEEDS, "--max-rows", "25000", "--output", str(RERUNS / "qtransport")]),
    "matched-control": ("CODE/experiment_low_label/run_matched_expert_control.py", ["--source-seeds", Q_SEEDS, "--max-rows", "25000", "--output", str(RERUNS / "matched_control")]),
    "q-summary": ("CODE/experiment_low_label/derive_rq3_seed_basis_9.py", ["--output", str(RERUNS / "q_summary")]),
    "q-check": ("CODE/experiment_low_label/check_qtransport_reproduction.py", []),
    "staircase": ("CODE/experiment_staircase/run_s0_s3_staircase.py", ["--out", str(RERUNS / "staircase"), "--score-root", str(RERUNS / "stage3")]),
    "low-budget": ("CODE/experiment_low_label/run_nfv2_target_label_budget_confirmation_v1.py", ["--output-root", str(RERUNS / "low_budget"), "--stage3-root", str(RERUNS / "stage3")]),
    "adaptation": ("CODE/experiment_low_label/run_finetune_from_checkpoint.py", ["--output_root", str(RERUNS / "adaptation")]),
    "deployment-summary": ("CODE/experiment_low_label/aggregate_nfv2_target_label_budget_confirmation_v1.py", ["--output-root", str(RERUNS / "deployment_summary")]),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script, defaults = COMMANDS[args.experiment]
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, str(ROOT / script), *defaults, *[s for s in extra if s != "--print-command"]]
    if "--print-command" in extra:
        import shlex
        print(shlex.join(command))
        return 0
    return subprocess.call(command, cwd=ROOT)

if __name__ == "__main__":
    raise SystemExit(main())
