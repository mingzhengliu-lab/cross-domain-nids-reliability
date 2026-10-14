import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))
"""
Aggregate ``results/stage3/*/final/summary.json`` into publication tables.

Covers 28 (method, target) cells for the main DG benchmark and writes:
    Table_3_main.md / .tex
    Table_3_raw.csv
    summary_analysis.md

Usage:
    python CODE/run_paper.py stage3-summary
    python CODE/run_paper.py stage3-summary --output_dir RESULTS/reruns/stage3_summary
"""
import argparse
import csv
import json
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from nidsbed.algorithm_metadata import get_algorithm_metadata, method_display_name

DEFAULT_METHODS_ORDER = ['erm', 'irm', 'coral', 'groupdro', 'dann', 'fishr', 'boda']
TARGETS_ORDER = ['unsw', 'cic', 'bot', 'ton']
METHOD_DISPLAY = {
    method: method_display_name(method)
    for method in DEFAULT_METHODS_ORDER
}
TARGET_DISPLAY = {
    'unsw': 'UNSW',
    'cic': 'CIC',
    'bot': 'BoT',
    'ton': 'ToN',
}


def method_order(rows):
    seen = {r['method'] for r in rows}
    ordered = [m for m in DEFAULT_METHODS_ORDER if m in seen]
    extras = sorted(m for m in seen if m not in DEFAULT_METHODS_ORDER)
    return ordered + extras


def load_all(stage3_dir):
    rows = []
    pattern = os.path.join(stage3_dir, '*/final/summary.json')
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        combo = os.path.basename(os.path.dirname(os.path.dirname(path)))
        method, target = combo.split('_', 1)
        metadata = d.get('algorithm_metadata') or get_algorithm_metadata(method)
        rows.append({
            'method': method,
            'method_display': metadata.get('display_name', METHOD_DISPLAY.get(method, method)),
            'implementation_variant': metadata.get('implementation_variant', ''),
            'fidelity': metadata.get('fidelity', ''),
            'target': target,
            'hp_id': d.get('hp_id'),
            'seeds': d.get('seeds'),
            'search_seeds': d.get('search_seeds'),
            'selection_metric': d.get('selection_metric'),
            'early_stop_metric': d.get('early_stop_metric'),
            'in_f1_mean': d['in_attack_f1_mean'],
            'in_f1_std': d['in_attack_f1_std'],
            'cross_f1_mean': d['cross_attack_f1_mean'],
            'cross_f1_std': d['cross_attack_f1_std'],
            'in_auroc_mean': d['in_auroc_mean'],
            'in_auroc_std': d['in_auroc_std'],
            'cross_auroc_mean': d['cross_auroc_mean'],
            'cross_auroc_std': d['cross_auroc_std'],
            'in_auprc_mean': d.get('in_auprc_mean'),
            'in_auprc_std': d.get('in_auprc_std'),
            'cross_auprc_mean': d.get('cross_auprc_mean'),
            'cross_auprc_std': d.get('cross_auprc_std'),
            'cross_tpr_at_fpr_1pct_mean': d.get('cross_tpr_at_fpr_1pct_mean'),
            'cross_tpr_at_fpr_1pct_std': d.get('cross_tpr_at_fpr_1pct_std'),
            'cross_unique_attack_f1_mean': d.get('cross_unique_attack_f1_mean'),
            'cross_unique_attack_f1_std': d.get('cross_unique_attack_f1_std'),
            'cross_unique_auroc_mean': d.get('cross_unique_auroc_mean'),
            'cross_unique_auroc_std': d.get('cross_unique_auroc_std'),
            'cross_unique_auprc_mean': d.get('cross_unique_auprc_mean'),
            'cross_unique_auprc_std': d.get('cross_unique_auprc_std'),
            'in_precision_mean': d.get('in_precision_mean'),
            'cross_precision_mean': d.get('cross_precision_mean'),
            'in_recall_mean': d.get('in_recall_mean'),
            'cross_recall_mean': d.get('cross_recall_mean'),
            'degradation_mean': d['degradation_mean'],
            'degradation_std': d['degradation_std'],
        })
    expected = len(DEFAULT_METHODS_ORDER) * len(TARGETS_ORDER)
    if len(rows) != expected:
        print(f"[warn] found {len(rows)} combos, baseline benchmark expects {expected}")
    return rows


def by_method_target(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r['method']][r['target']] = r
    return d


def write_csv(rows, path):
    keys = ['method', 'method_display', 'implementation_variant', 'fidelity',
            'target', 'hp_id', 'seeds', 'search_seeds',
            'selection_metric', 'early_stop_metric',
            'in_f1_mean', 'in_f1_std',
            'cross_f1_mean', 'cross_f1_std',
            'in_auroc_mean', 'in_auroc_std',
            'cross_auroc_mean', 'cross_auroc_std',
            'in_auprc_mean', 'in_auprc_std',
            'cross_auprc_mean', 'cross_auprc_std',
            'cross_tpr_at_fpr_1pct_mean', 'cross_tpr_at_fpr_1pct_std',
            'cross_unique_attack_f1_mean', 'cross_unique_attack_f1_std',
            'cross_unique_auroc_mean', 'cross_unique_auroc_std',
            'cross_unique_auprc_mean', 'cross_unique_auprc_std',
            'in_precision_mean', 'cross_precision_mean',
            'in_recall_mean', 'cross_recall_mean',
            'degradation_mean', 'degradation_std']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  [OK] {path}")


def write_markdown(rows, path):
    bm = by_method_target(rows)
    methods = method_order(rows)
    final_seeds = next((r.get('seeds') for r in rows if r.get('seeds')), [2026, 2027, 2028])
    search_seeds = next((r.get('search_seeds') for r in rows if r.get('search_seeds')), [42, 123])
    selection_metric = next((r.get('selection_metric') for r in rows if r.get('selection_metric')),
                            'val_auroc_macro')
    early_stop_metric = next((r.get('early_stop_metric') for r in rows if r.get('early_stop_metric')),
                             selection_metric)
    lines = []
    lines.append(f"# Main DG Benchmark Results ({len(rows)} combos)")
    lines.append("")
    lines.append(f"Each cell shows `mean +/- std` over final seeds {final_seeds}.")
    lines.append(f"HP search uses search seeds {search_seeds}; entries are ranked by `{selection_metric}`.")
    lines.append(f"Best epoch selection uses `{early_stop_metric}` on source validation.")
    lines.append("Method names are implementation-aware; see the variant notes below.")
    lines.append("")

    lines.append("## Method variant notes")
    lines.append("")
    lines.append("| Method key | Display | Implementation variant | Claim scope |")
    lines.append("|---|---|---|---|")
    for m in methods:
        meta = get_algorithm_metadata(m)
        lines.append(
            f"| `{m}` | {meta['display_name']} | "
            f"{meta['implementation_variant']} | {meta['claim_scope']} |"
        )
    lines.append("")

    # ----- Panel 1: cross_auroc (primary ranking metric) -----
    lines.append("## Panel A - `cross_auroc` (primary ranking metric)")
    lines.append("")
    header = "| Method | " + " | ".join(TARGET_DISPLAY[t] for t in TARGETS_ORDER) + " | Avg |"
    sep = "|--------|" + "|".join(["--------"] * (len(TARGETS_ORDER) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        cells = []
        vals = []
        for t in TARGETS_ORDER:
            r = bm.get(m, {}).get(t)
            if r is None:
                cells.append('--')
            else:
                vals.append(r['cross_auroc_mean'])
                marker = '*' if r['cross_auroc_mean'] < 0.5 else ''
                cells.append(f"{r['cross_auroc_mean']:.4f}{marker} +/- {r['cross_auroc_std']:.3f}")
        avg = sum(vals) / len(vals) if vals else 0
        row = f"| {method_display_name(m)} | " + " | ".join(cells) + f" | {avg:.4f} |"
        lines.append(row)
    # per-target avg
    avg_cells = []
    all_vals = []
    for t in TARGETS_ORDER:
        vs = [bm[m][t]['cross_auroc_mean'] for m in methods if t in bm.get(m, {})]
        a = sum(vs) / len(vs) if vs else 0
        all_vals.append(a)
        avg_cells.append(f"**{a:.4f}**")
    grand = sum(all_vals) / len(all_vals) if all_vals else 0
    lines.append(f"| **Avg** | " + " | ".join(avg_cells) + f" | **{grand:.4f}** |")
    lines.append("")

    # ----- Panel 2: cross_auprc (imbalance-sensitive ranking metric) -----
    lines.append("## Panel B - `cross_auprc` (imbalance-sensitive ranking metric)")
    lines.append("")
    lines.append("PR-AUC is sensitive to the target attack prior, so compare it with AUROC and attack rate.")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in methods:
        cells = []
        vals = []
        for t in TARGETS_ORDER:
            r = bm.get(m, {}).get(t)
            if r is None:
                cells.append('--')
            else:
                if r.get('cross_auprc_mean') is None:
                    cells.append('--')
                    continue
                vals.append(r['cross_auprc_mean'])
                cells.append(f"{r['cross_auprc_mean']:.4f} +/- {r.get('cross_auprc_std') or 0:.3f}")
        avg = sum(vals) / len(vals) if vals else 0
        row = f"| {method_display_name(m)} | " + " | ".join(cells) + f" | {avg:.4f} |"
        lines.append(row)
    avg_cells = []
    all_vals = []
    for t in TARGETS_ORDER:
        vs = [bm[m][t]['cross_auprc_mean'] for m in methods
              if t in bm.get(m, {}) and bm[m][t].get('cross_auprc_mean') is not None]
        a = sum(vs) / len(vs) if vs else 0
        all_vals.append(a)
        avg_cells.append(f"**{a:.4f}**")
    grand = sum(all_vals) / len(all_vals) if all_vals else 0
    lines.append(f"| **Avg** | " + " | ".join(avg_cells) + f" | **{grand:.4f}** |")
    lines.append("")

    # ----- Panel 3: fixed-threshold F1 diagnostic -----
    lines.append("## Panel C - `cross_attack_f1` (fixed-threshold diagnostic)")
    lines.append("")
    lines.append("This is the argmax/0.5-threshold F1. Treat it as a threshold/calibration diagnostic, not the primary deployment metric.")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in methods:
        cells = []
        vals = []
        for t in TARGETS_ORDER:
            r = bm.get(m, {}).get(t)
            if r is None:
                cells.append('--')
            else:
                vals.append(r['cross_f1_mean'])
                cells.append(f"{r['cross_f1_mean']:.4f} +/- {r['cross_f1_std']:.3f}")
        avg = sum(vals) / len(vals) if vals else 0
        row = f"| {method_display_name(m)} | " + " | ".join(cells) + f" | {avg:.4f} |"
        lines.append(row)
    lines.append("")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] {path}")


def write_latex(rows, path):
    """Write a LaTeX table aligned with the Markdown main panels."""
    bm = by_method_target(rows)
    methods = method_order(rows)
    lines = []
    lines.append(r"% Auto-generated by stage3_aggregate.py - edit at your own risk")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Main benchmark results across the available DG methods and 4 LODO targets. Each cell reports mean $\pm$ std over final seeds. \textbf{Bold}: best per column. $^\dagger$: representation inversion ($\mathrm{AUROC}_{\mathrm{cross}} < 0.5$). Fixed-threshold F1 is reported as a diagnostic.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{lcccc|c}")
    lines.append(r"\toprule")
    lines.append(r"Method & UNSW & CIC & BoT & ToN & Avg \\")
    panels = [
        (
            r"Panel A: $\mathrm{AUROC}_{\mathrm{cross}}$ (primary ranking metric)",
            "cross_auroc_mean",
            "cross_auroc_std",
            True,
        ),
        (
            r"Panel B: $\mathrm{AUPRC}_{\mathrm{cross}}$ (imbalance-sensitive ranking metric)",
            "cross_auprc_mean",
            "cross_auprc_std",
            False,
        ),
        (
            r"Panel C: fixed-threshold $\mathrm{F1}_{\mathrm{cross}}$ diagnostic",
            "cross_f1_mean",
            "cross_f1_std",
            False,
        ),
    ]

    for panel_idx, (panel_title, mean_key, std_key, mark_inversion) in enumerate(panels):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{c}}{{\textit{{{panel_title}}}}} \\")
        lines.append(r"\midrule")

        best_per_target = {}
        for t in TARGETS_ORDER:
            vs = [
                (m, bm[m][t].get(mean_key))
                for m in methods
                if t in bm.get(m, {}) and bm[m][t].get(mean_key) is not None
            ]
            vs = [(m, float(v)) for m, v in vs if not np.isnan(float(v))]
            best_per_target[t] = max(vs, key=lambda x: x[1])[0] if vs else None

        for m in methods:
            cells = []
            vals = []
            for t in TARGETS_ORDER:
                r = bm.get(m, {}).get(t)
                value = None if r is None else r.get(mean_key)
                if value is None or np.isnan(float(value)):
                    cells.append('--')
                    continue
                value = float(value)
                vals.append(value)
                std = r.get(std_key) or 0.0
                dag = r"^\dagger" if mark_inversion and value < 0.5 else ""
                s = f"${value:.3f}{dag}{{\\scriptstyle\\pm{float(std):.3f}}}$"
                if best_per_target[t] == m:
                    s = r"\textbf{" + s + "}"
                cells.append(s)
            avg = sum(vals) / len(vals) if vals else float('nan')
            avg_s = "--" if np.isnan(avg) else f"${avg:.3f}$"
            lines.append(f"{method_display_name(m)} & " + " & ".join(cells) + f" & {avg_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] {path}")


def write_findings_md(rows, attack_rates, corr_info, path):
    """Write the per-target / per-method findings markdown summary."""
    bm = by_method_target(rows)

    # gather stats
    all_au = [r['cross_auroc_mean'] for r in rows]
    inversion_n = sum(1 for x in all_au if x < 0.5)
    total = len(rows)

    lines = []
    lines.append("# Stage 3 - Key Findings Summary")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- **Representation inversion prevalence**: {inversion_n}/{total} = "
                 f"{100*inversion_n/total:.0f}% of (method, target) combinations have cross_auroc < 0.5")
    if corr_info.get("n", 0) >= 2 and np.isfinite(corr_info.get("r", float("nan"))):
        lines.append(f"- **Strongest predictor**: signed prior shift (target - source mean attack rate) "
                     f"explains {corr_info['r']**2*100:.0f}% of cross_auroc variance")
        lines.append(f"  (Pearson r = {corr_info['r']:+.3f}, p = {corr_info['p']:.2e})")
    else:
        lines.append("- **Prior-shift correlation**: not computed because fewer than two "
                     "Stage-3 points are available in this aggregate.")
    lines.append("")

    # domain attack rates
    lines.append("## Label prior composition")
    lines.append("")
    lines.append("| Domain | Attack rate | Prior shift (vs source mean) |")
    lines.append("|--------|-------------|------------------------------|")
    domains = list(attack_rates.keys())
    for dom in TARGETS_ORDER:
        r = attack_rates[dom]
        src_domains = [s for s in domains if s != dom]
        src_mean = sum(attack_rates[s] for s in src_domains) / len(src_domains)
        shift = r - src_mean
        lines.append(f"| {TARGET_DISPLAY[dom]} | {r*100:.2f}% | {shift*100:+.2f}% |")
    lines.append("")

    # real rescues (find combos with cross_auroc > 0.5 AND std < 0.05)
    lines.append("## Real rescues (cross_auroc > 0.5 AND std < 0.05, 3 seeds)")
    lines.append("")
    lines.append("| Method x Target | cross_auroc | cross_f1 | in_f1 | verdict |")
    lines.append("|-----------------|-------------|----------|-------|---------|")
    for r in rows:
        if r['cross_auroc_mean'] > 0.5 and r['cross_auroc_std'] < 0.05:
            lines.append(f"| {method_display_name(r['method'])} x {TARGET_DISPLAY[r['target']]} "
                         f"| {r['cross_auroc_mean']:.4f} +/- {r['cross_auroc_std']:.3f} "
                         f"| {r['cross_f1_mean']:.4f} +/- {r['cross_f1_std']:.3f} "
                         f"| {r['in_f1_mean']:.4f} | stable OK |")
    lines.append("")

    # lucky wins (high mean but high std)
    lines.append("## Lucky wins (high mean cross_f1 but CV > 0.3 - single-seed unreliability)")
    lines.append("")
    lines.append("| Method x Target | cross_f1 | CV | seed values |")
    lines.append("|-----------------|----------|----|-----|")
    for r in rows:
        if r['cross_f1_mean'] > 0.1 and r['cross_f1_std'] / max(r['cross_f1_mean'], 1e-9) > 0.3:
            # load per-seed to show them
            p = f"results/stage3/{r['method']}_{r['target']}/final/summary.json"
            if os.path.exists(p):
                d = json.load(open(p))
                seeds = [f"{s['cross_attack_f1']:.3f}" for s in d.get('per_seed', [])]
            else:
                seeds = []
            cv = r['cross_f1_std'] / r['cross_f1_mean']
            lines.append(f"| {method_display_name(r['method'])} x {TARGET_DISPLAY[r['target']]} "
                         f"| {r['cross_f1_mean']:.3f} +/- {r['cross_f1_std']:.3f} "
                         f"| {cv:.2f} "
                         f"| [{', '.join(seeds)}] |")
    lines.append("")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] {path}")


def compute_attack_rates_and_corr(rows, data_dir):
    import numpy as np
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("[warn] scipy not available; correlations omitted")
        return {}, {'r': 0, 'p': 1}

    attack_rates = {}
    for dom in TARGETS_ORDER:
        y = np.load(os.path.join(data_dir, f"{dom}_y.npy"))
        attack_rates[dom] = float(y.mean())

    xs, ys = [], []
    for r in rows:
        src_domains = [s for s in TARGETS_ORDER if s != r['target']]
        src_mean = sum(attack_rates[s] for s in src_domains) / len(src_domains)
        signed = attack_rates[r['target']] - src_mean
        xs.append(signed)
        ys.append(r['cross_auroc_mean'])

    finite = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))
    ]
    if len(finite) < 2:
        return attack_rates, {
            'r': float('nan'),
            'p': float('nan'),
            'rho': float('nan'),
            'p_spearman': float('nan'),
            'n': len(finite),
        }
    xs, ys = zip(*finite)
    r_p, p_p = pearsonr(xs, ys)
    r_s, p_s = spearmanr(xs, ys)
    return attack_rates, {'r': r_p, 'p': p_p, 'rho': r_s, 'p_spearman': p_s, 'n': len(finite)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3_dir', default='RESULTS/stage3_lodo/stage3')
    parser.add_argument('--data_dir', default='DATA/nfv2_processed')
    parser.add_argument('--output_dir', default='RESULTS/stage3_lodo/aggregate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Aggregating {args.stage3_dir} -> {args.output_dir}")
    print()

    rows = load_all(args.stage3_dir)
    print(f"Loaded {len(rows)} combos")
    print()
    if not rows:
        raise SystemExit(
            f"No Stage-3 summaries found under {args.stage3_dir}. "
            "Run Stage-3 first, or resume the experiment before aggregating."
        )

    attack_rates, corr = compute_attack_rates_and_corr(rows, args.data_dir)

    print("Writing outputs:")
    write_csv(rows, os.path.join(args.output_dir, 'Table_3_raw.csv'))
    write_markdown(rows, os.path.join(args.output_dir, 'Table_3_main.md'))
    write_latex(rows, os.path.join(args.output_dir, 'Table_3_main.tex'))
    if attack_rates:
        write_findings_md(rows, attack_rates, corr,
                          os.path.join(args.output_dir, 'summary_analysis.md'))

    print()
    print(f"Done. See {args.output_dir}/")


if __name__ == '__main__':
    main()
