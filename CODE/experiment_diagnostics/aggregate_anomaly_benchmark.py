"""
Aggregate the source-benign-only anomaly benchmark for the active project.

Aggregate scratch/exp03_model_family_controls/anomaly_runs/<detector>_<target>/seed*.json into:
  - Table_anomaly_main.md   (Panels A-D: AUROC + F1 at 3 thresholds + in-domain sanity)
  - Table_anomaly_main.tex  (booktabs LaTeX, experiment-ready)
  - Table_anomaly_raw.csv   (flat dump of per-run records)
  - summary_anomaly.md      (head-to-head with the strict-DG ERM baseline)

Comparison numbers from the main benchmark are read from the active
  RESULTS/stage3_lodo/stage3_aggregate/Table_3_raw.csv (per-method per-target).

Usage:
    python CODE/experiment_diagnostics/aggregate_anomaly_benchmark.py
    python CODE/experiment_diagnostics/aggregate_anomaly_benchmark.py --anomaly_dir /path/to/anomaly_runs
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = PROJECT_ROOT / 'RESULTS' / 'model_family'


DETECTORS_ORDER = ['mahalanobis', 'iforest', 'autoencoder', 'deep_svdd']
TARGETS_ORDER   = ['unsw', 'cic', 'bot', 'ton']
DETECTOR_DISPLAY = {
    'mahalanobis': 'Mahal.',
    'iforest':     'IForest',
    'autoencoder': 'AE',
    'deep_svdd':   'DSVDD',
}
TARGET_DISPLAY = {'unsw': 'UNSW', 'cic': 'CIC', 'bot': 'BoT', 'ton': 'ToN'}


# -----------------------------------------------------------------------------
# Collect & aggregate
# -----------------------------------------------------------------------------
def collect(anomaly_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(anomaly_dir, '*', 'seed*.json'))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def aggregate_by_combo(rows):
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[(r['detector'], r['target'])].append(r)

    def sample_std(values):
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    out = {}
    for k, group in by_combo.items():
        cf1_p95    = [g['cross']['tau_p95']['attack_f1']    for g in group]
        cf1_p99    = [g['cross']['tau_p99']['attack_f1']    for g in group]
        cf1_oracle = [g['cross']['tau_oracle']['attack_f1'] for g in group]
        cau        = [g['cross']['auroc'] for g in group if g['cross']['auroc'] is not None]
        if1_p95    = [g['in']['tau_p95']['attack_f1'] for g in group]
        iau        = [g['in']['auroc']    for g in group if g['in']['auroc'] is not None]
        out[k] = {
            'n_seeds':              len(group),
            'cross_f1_p95_mean':    float(np.mean(cf1_p95)),
            'cross_f1_p95_std':     sample_std(cf1_p95),
            'cross_f1_p99_mean':    float(np.mean(cf1_p99)),
            'cross_f1_p99_std':     sample_std(cf1_p99),
            'cross_f1_oracle_mean': float(np.mean(cf1_oracle)),
            'cross_f1_oracle_std':  sample_std(cf1_oracle),
            'cross_auroc_mean':     float(np.mean(cau)) if cau else None,
            'cross_auroc_std':      sample_std(cau) if cau else None,
            'in_f1_p95_mean':       float(np.mean(if1_p95)),
            'in_f1_p95_std':        sample_std(if1_p95),
            'in_auroc_mean':        float(np.mean(iau)) if iau else None,
            'target_attack_rate':   float(np.mean([g['target_attack_rate'] for g in group])),
        }
    return out


def load_erm_baseline(stage3_csv):
    if not Path(stage3_csv).exists():
        return None
    out = {}
    with open(stage3_csv) as f:
        for row in csv.DictReader(f):
            if row.get('method') == 'erm':
                out[row['target']] = {
                    'cross_f1_mean':    float(row['cross_f1_mean']),
                    'cross_auroc_mean': float(row['cross_auroc_mean']),
                }
    return out


def load_delta_pi(data_dir=PROJECT_ROOT / 'DATA' / 'nfv2_processed'):
    """Compute target prior shift from processed labels if available."""
    rates = {}
    for target in TARGETS_ORDER:
        path = Path(data_dir) / f'{target}_y.npy'
        if not path.exists():
            return {}
        rates[target] = float(np.mean(np.load(path, mmap_mode='r')))
    return {
        target: rates[target] - float(np.mean([rates[t] for t in TARGETS_ORDER if t != target]))
        for target in TARGETS_ORDER
    }


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def write_csv(rows, path):
    if not rows:
        return
    flat = []
    for r in rows:
        f = {}
        for k, v in r.items():
            if k == 'env':
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for kkk, vvv in vv.items():
                            f[f'{k}.{kk}.{kkk}'] = vvv
                    else:
                        f[f'{k}.{kk}'] = vv
            else:
                f[k] = v
        flat.append(f)
    keys = []
    for r in flat:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in flat:
            w.writerow(r)


def _panel(lines, title, metric_key, by_combo, std_key=None,
           erm_baseline=None, erm_metric='cross_f1_mean'):
    lines.append(f"## {title}\n")
    header = "| Detector | " + " | ".join(TARGET_DISPLAY[t] for t in TARGETS_ORDER) + " | Avg |"
    sep    = "|" + "---|" * (len(TARGETS_ORDER) + 2)
    lines.append(header); lines.append(sep)
    for det in DETECTORS_ORDER:
        cells, vals = [], []
        for t in TARGETS_ORDER:
            r = by_combo.get((det, t))
            if r is None or r[metric_key] is None:
                cells.append('--')
            else:
                vals.append(r[metric_key])
                if std_key and r.get(std_key) is not None:
                    cells.append(f"{r[metric_key]:.3f} +/- {r[std_key]:.3f}")
                else:
                    cells.append(f"{r[metric_key]:.3f}")
        avg = sum(vals) / len(vals) if vals else 0
        lines.append(f"| {DETECTOR_DISPLAY[det]} | " + " | ".join(cells) + f" | {avg:.3f} |")
    if erm_baseline:
        cells = []
        vals = []
        for t in TARGETS_ORDER:
            if t in erm_baseline and erm_baseline[t].get(erm_metric) is not None:
                vals.append(erm_baseline[t][erm_metric])
                cells.append(f"{erm_baseline[t][erm_metric]:.3f}")
            else:
                cells.append('--')
        avg = sum(vals) / len(vals) if vals else 0
        lines.append(f"| _ERM (main, ref)_ | " + " | ".join(cells) + f" | _{avg:.3f}_ |")
    lines.append("")


def write_markdown(by_combo, path, erm_baseline=None):
    lines = []
    lines.append("# Anomaly Detection Benchmark")
    lines.append("")
    lines.append("Source-benign-only training. Strict P0 protocol (no source-attack at fit, "
                 "no target X at fit). 4 detectors x 4 targets x 3 seeds = 48 runs.")
    lines.append("")
    lines.append("**Threshold rules**:")
    lines.append("- `tau_p95` / `tau_p99`: q-quantile of source benign training scores. "
                 "Uses no attack info, no target info.")
    lines.append("- `tau_oracle`: best F1 on target via grid search. Upper bound for threshold selection.")
    lines.append("")

    _panel(lines, "Panel A - Cross-domain AUROC (primary, threshold-free)",
           'cross_auroc_mean', by_combo, std_key='cross_auroc_std',
           erm_baseline=erm_baseline, erm_metric='cross_auroc_mean')

    _panel(lines, "Panel B - Cross-domain Attack-F1 at tau_p95",
           'cross_f1_p95_mean', by_combo, std_key='cross_f1_p95_std',
           erm_baseline=erm_baseline, erm_metric='cross_f1_mean')

    _panel(lines, "Panel C - Cross-domain Attack-F1 at tau_oracle (target upper bound)",
           'cross_f1_oracle_mean', by_combo, std_key='cross_f1_oracle_std')

    _panel(lines, "Panel D - In-domain AUROC (sanity)",
           'in_auroc_mean', by_combo)

    Path(path).write_text('\n'.join(lines))


def write_latex(by_combo, path, erm_baseline=None):
    lines = []
    lines.append(r"% Auto-generated by anomaly_aggregate.py")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Source-benign-only anomaly detection benchmark. "
                 r"Each cell reports mean $\pm$ std over 3 seeds $\{2026,2027,2028\}$. "
                 r"Strict P0 protocol; no source-attack labels and no target $X$ at fit time. "
                 r"Last row: ERM main benchmark for reference.}")
    lines.append(r"\label{tab:anomaly}")
    lines.append(r"\begin{tabular}{lcccc|c}")
    lines.append(r"\toprule")
    lines.append(r"Detector & UNSW & CIC & BoT & ToN & Avg \\")
    lines.append(r"\midrule")

    def panel(title, key_mean, key_std, with_erm_metric):
        lines.append(rf"\multicolumn{{6}}{{c}}{{\textit{{{title}}}}} \\")
        lines.append(r"\midrule")
        for det in DETECTORS_ORDER:
            cells, vals = [], []
            for t in TARGETS_ORDER:
                r = by_combo.get((det, t))
                if r is None or r[key_mean] is None:
                    cells.append('--')
                else:
                    vals.append(r[key_mean])
                    cells.append(f"${r[key_mean]:.3f}{{\\scriptstyle\\pm{r[key_std]:.3f}}}$")
            avg = sum(vals) / len(vals) if vals else 0
            lines.append(f"{DETECTOR_DISPLAY[det]} & " + " & ".join(cells) + f" & ${avg:.3f}$ \\\\")
        if erm_baseline and with_erm_metric:
            cells, vals = [], []
            for t in TARGETS_ORDER:
                v = erm_baseline.get(t, {}).get(with_erm_metric)
                if v is None:
                    cells.append('--')
                else:
                    vals.append(v)
                    cells.append(f"${v:.3f}$")
            avg = sum(vals) / len(vals) if vals else 0
            lines.append(r"\textit{ERM (ref)} & " + " & ".join(cells) + f" & $\\textit{{{avg:.3f}}}$ \\\\")

    panel(r"Panel A: $\mathrm{AUROC}_{\mathrm{cross}}$",
          'cross_auroc_mean', 'cross_auroc_std', 'cross_auroc_mean')
    lines.append(r"\midrule")
    panel(r"Panel B: $\mathrm{F1}_{\mathrm{cross}}$ at $\tau_{p95}$",
          'cross_f1_p95_mean', 'cross_f1_p95_std', 'cross_f1_mean')
    lines.append(r"\midrule")
    panel(r"Panel C: $\mathrm{F1}_{\mathrm{cross}}$ at $\tau_{\mathrm{oracle}}$ (target upper bound)",
          'cross_f1_oracle_mean', 'cross_f1_oracle_std', None)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    Path(path).write_text('\n'.join(lines))


def write_summary(by_combo, path, erm_baseline=None):
    lines = []
    lines.append("# Anomaly Detection - Key Findings\n")
    lines.append("Source-benign-only (P0). Comparison columns are read from the ERM "
                 "main benchmark when available.\n")

    lines.append("## Headline numbers per target\n")
    delta_pi = load_delta_pi()
    lines.append("| Target | d_pi | Best detector by AUROC | AUROC | "
                 "F1@p95 | F1@oracle | ERM cross-F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in TARGETS_ORDER:
        # find best detector by mean cross AUROC
        best = None
        best_au = -1
        for det in DETECTORS_ORDER:
            r = by_combo.get((det, t))
            if r and r['cross_auroc_mean'] is not None and r['cross_auroc_mean'] > best_au:
                best_au = r['cross_auroc_mean']
                best = (det, r)
        if best is None:
            continue
        det, r = best
        erm = (erm_baseline or {}).get(t, {})
        dpi = delta_pi.get(t)
        dpi_text = f"{dpi * 100:+.1f}%" if dpi is not None else "n/a"
        lines.append(
            f"| {TARGET_DISPLAY[t]} "
            f"| {dpi_text} "
            f"| {DETECTOR_DISPLAY[det]} "
            f"| {r['cross_auroc_mean']:.3f} +/- {r['cross_auroc_std']:.3f} "
            f"| {r['cross_f1_p95_mean']:.3f} +/- {r['cross_f1_p95_std']:.3f} "
            f"| {r['cross_f1_oracle_mean']:.3f} +/- {r['cross_f1_oracle_std']:.3f} "
            f"| {erm.get('cross_f1_mean', float('nan')):.3f} |"
        )
    lines.append("")

    # Did anomaly detection win on AUROC anywhere ERM was inverted?
    lines.append("## Where anomaly detection beats ERM on cross-AUROC\n")
    lines.append("| Detector | Target | Anomaly AUROC | ERM AUROC | Δ |")
    lines.append("|---|---|---|---|---|")
    if erm_baseline:
        for det in DETECTORS_ORDER:
            for t in TARGETS_ORDER:
                r = by_combo.get((det, t))
                if r is None or r['cross_auroc_mean'] is None:
                    continue
                erm_au = erm_baseline.get(t, {}).get('cross_auroc_mean')
                if erm_au is None:
                    continue
                delta = r['cross_auroc_mean'] - erm_au
                if delta > 0.02:
                    lines.append(f"| {DETECTOR_DISPLAY[det]} | {TARGET_DISPLAY[t]} "
                                 f"| {r['cross_auroc_mean']:.3f} | {erm_au:.3f} "
                                 f"| **+{delta:.3f}** |")
    lines.append("")

    Path(path).write_text('\n'.join(lines))


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_dir', default=str(SCRATCH_ROOT / 'anomaly'))
    parser.add_argument('--out_dir',     default=str(SCRATCH_ROOT / 'anomaly_aggregate'))
    parser.add_argument('--stage3_csv',  default=str(PROJECT_ROOT / 'RESULTS' / 'stage3_lodo' / 'aggregate' / 'Table_3_raw.csv'))
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows = collect(args.anomaly_dir)
    print(f"loaded {len(rows)} runs from {args.anomaly_dir}")
    if not rows:
        print("no runs found; nothing to aggregate")
        return
    by_combo = aggregate_by_combo(rows)
    print(f"aggregated to {len(by_combo)} (detector, target) combos")

    erm = load_erm_baseline(args.stage3_csv)
    if erm:
        print(f"ERM baseline loaded: {sorted(erm.keys())}")
    else:
        print(f"[note] ERM baseline not found at {args.stage3_csv}")

    write_csv(rows,        os.path.join(args.out_dir, 'Table_anomaly_raw.csv'))
    write_markdown(by_combo, os.path.join(args.out_dir, 'Table_anomaly_main.md'),
                   erm_baseline=erm)
    write_latex(by_combo,  os.path.join(args.out_dir, 'Table_anomaly_main.tex'),
                erm_baseline=erm)
    write_summary(by_combo, os.path.join(args.out_dir, 'summary_anomaly.md'),
                  erm_baseline=erm)

    print(f"\nWrote:")
    for f in ['Table_anomaly_main.md', 'Table_anomaly_main.tex',
              'Table_anomaly_raw.csv', 'summary_anomaly.md']:
        print(f"  {os.path.join(args.out_dir, f)}")


if __name__ == '__main__':
    main()
