"""Render Figures 4-9 from the distributed experiment records.

Outputs default to RESULTS/reruns/figures. Published paper assets and archived
metrics are read-only. No traffic arrays, checkpoints, or model training needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from CODE.analysis.output_paths import output_directory
from CODE.analysis.paper_results import (
    RQ1, Q, CONTROL, STAIR, DEPLOY, DOMAINS, DISPLAY, METHODS, METHOD_DISPLAY,
    SEEDS, SUPPORT_SEEDS,
)

COLORS = dict(zip(DOMAINS, ("#3567a8", "#1b9e77", "#d89000", "#c44e52")))
STEMS = {
    4: "fig4_target_group_auroc_heatmap",
    5: "fig5_source_target_contrast",
    6: "fig6_row_group_estimand",
    7: "fig7_qtransport_paired_deltas",
    8: "fig8_staircase_heatmap",
    9: "fig9_deployment_operating_points",
}
INPUTS = set()


def read_csv(path):
    INPUTS.add(path)
    return pd.read_csv(path)


def read_json(path):
    INPUTS.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main_matrix():
    frame = read_csv(RQ1 / "Table_3_raw.csv")
    # Three-seed means are the plotted unit. Check them against the per-seed data.
    columns = ("in_auroc", "cross_auroc", "cross_unique_auroc")
    for row in frame.itertuples():
        records = [read_json(RQ1 / f"per_cell/{row.method}_{row.target}/seed{s}.json")
                   for s in (2026, 2027, 2028)]
        for column in columns:
            mean = np.mean([record[column] for record in records])
            if not np.isclose(mean, getattr(row, column + "_mean"), atol=1e-12, rtol=0):
                raise ValueError(f"Archived seed mean differs: {row.method}/{row.target}/{column}")
    return frame


def heatmap(ax, values, xlabels, ylabels, vmin, vmax=1):
    mesh = ax.pcolormesh(values, cmap="BrBG", norm=TwoSlopeNorm(vmin=vmin, vcenter=.5, vmax=vmax),
                         edgecolors="white", linewidth=.7, shading="flat")
    ax.set_xticks(np.arange(len(xlabels)) + .5, xlabels)
    ax.set_yticks(np.arange(len(ylabels)) + .5, ylabels)
    ax.invert_yaxis()
    ax.tick_params(length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), value in np.ndenumerate(values):
        rgb = mesh.cmap(mesh.norm(value))[:3]
        color = "white" if np.dot(rgb, [.299, .587, .114]) < .57 else "#262626"
        ax.text(j + .5, i + .5, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
    return mesh


def figure4():
    frame = main_matrix()
    values = frame.pivot(index="method", columns="target", values="cross_unique_auroc_mean").loc[list(METHODS), list(DOMAINS)]
    fig, ax = plt.subplots(figsize=(6.92, 4.20))
    fig.subplots_adjust(left=.19, right=.87, bottom=.23, top=.98)
    mesh = heatmap(ax, values.to_numpy(), [DISPLAY[t] for t in DOMAINS], [METHOD_DISPLAY[m] for m in METHODS], 0)
    ax.set(xlabel="Held-out target domain", ylabel="Source-trained method")
    cax = fig.add_axes([.895, .30, .022, .61])
    colorbar = fig.colorbar(mesh, cax=cax, label="Target group AUROC")
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    fig.text(.54, .045, "Center = chance level (0.5); each cell is the mean over three training seeds.", ha="center", fontsize=8, color="#666666")
    return fig, frame[["method", "target", "cross_unique_auroc_mean"]], "Three training seeds per method-target cell; duplicate-group AUROC."


def figure5():
    frame = main_matrix().sort_values(["target", "method"])
    domains = tuple(reversed(DOMAINS))
    fig, ax = plt.subplots(figsize=(509.146875 / 72, 371.390755 / 72))
    fig.subplots_adjust(left=.11, right=.985, bottom=.18, top=.90)
    ax.set(xlim=(0, 1.02), ylim=(-.45, 3.45), xlabel="Row-level AUROC", ylabel="Held-out target")
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_yticks(range(4), [DISPLAY[t] for t in domains])
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="x", color="#d0d0d0", linewidth=.9, zorder=0)
    for i, target in enumerate(domains):
        rows = frame[frame.target == target]
        for offset, row in zip(np.linspace(-.22, .22, 7), rows.itertuples()):
            y = i + offset
            line, = ax.plot([row.in_auroc_mean, row.cross_auroc_mean], [y, y], color="#a8a8a8", lw=1.05, zorder=2)
            line.set_gid(f"method_{target}_{row.method}")
            ax.scatter(row.in_auroc_mean, y, marker="o", s=25, facecolors="white", edgecolors=COLORS[target], linewidths=1.25, zorder=4)
            ax.scatter(row.cross_auroc_mean, y, marker="s", s=25, color=COLORS[target], edgecolors="white", linewidths=.4, zorder=4)
        source, target_row = rows.in_auroc_mean.mean(), rows.cross_auroc_mean.mean()
        line, = ax.plot([source, target_row], [i, i], color=COLORS[target], lw=3, alpha=.82, zorder=3)
        line.set_gid(f"mean_{target}")
        ax.scatter(source, i, marker="o", s=72, facecolors="white", edgecolors="#151515", linewidths=.9, zorder=6)
        ax.scatter(target_row, i, marker="s", s=72, color="#151515", linewidths=.6, zorder=6)
    ax.axvline(.5, color="#444444", ls="--", lw=1, zorder=5).set_gid("chance_0_5")
    ax.text(.505, 3.405, "chance", va="bottom", fontsize=9, color="#444444")
    handles = [Line2D([], [], marker="o", markersize=5, markerfacecolor="white", markeredgecolor="#262626", linestyle="none", label="Source-internal row"),
               Line2D([], [], marker="s", markersize=5, color="#262626", linestyle="none", label="Target row"),
               Line2D([], [], color="#777777", lw=1.5, label="Same method")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.035), ncol=3, frameon=False, fontsize=9.5, handlelength=1.5, columnspacing=1.8, handletextpad=.6, borderaxespad=0)
    fig.text(.56, .055, "Thin segments: seven methods, each averaged over three training seeds.", ha="center", fontsize=8.4, color="#555555")
    fig.text(.56, .021, "Thick segments and larger markers: mean across methods within each target.", ha="center", fontsize=8.4, color="#555555")
    values = frame[["method", "target", "in_auroc_mean", "cross_auroc_mean"]].copy()
    means = values.groupby("target", as_index=False)[["in_auroc_mean", "cross_auroc_mean"]].mean().assign(method="mean")
    return fig, pd.concat([values, means], ignore_index=True), "Both endpoints use row AUROC; three seeds per method, then an equal mean over seven methods."


def figure6():
    frame = main_matrix()
    fig, ax = plt.subplots(figsize=(5.01, 4.93))
    fig.subplots_adjust(left=.16, right=.98, bottom=.15, top=.98)
    for target in DOMAINS:
        rows = frame[frame.target == target]
        ax.scatter(rows.cross_auroc_mean, rows.cross_unique_auroc_mean, s=48, color=COLORS[target], alpha=.9, edgecolors="white", linewidths=.65, label=DISPLAY[target], zorder=3)
    ax.plot([.08, .72], [.08, .72], ":", color="#888888", lw=1.2, label="Equal estimands")
    ax.axhline(.5, color="#555555", ls="--", lw=1)
    ax.axvline(.5, color="#555555", ls="--", lw=1)
    ax.set(xlim=(.08, .72), ylim=(.08, .72), xlabel="Row-level target AUROC", ylabel="Duplicate-group target AUROC")
    ax.set_xticks(np.arange(.1, .71, .1)); ax.set_yticks(np.arange(.1, .71, .1))
    ax.set_aspect("equal"); ax.grid(color="#cccccc", lw=.8)
    for method, target, label, textxy in [("coral", "cic", "CORAL / CIC", (.14, .585)), ("dann", "ton", "DANN-source / ToN", (.34, .66))]:
        row = frame[(frame.method == method) & (frame.target == target)].iloc[0]
        ax.annotate(label, (row.cross_auroc_mean, row.cross_unique_auroc_mean), xytext=textxy, fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#777777", lw=1))
    ax.legend(loc="lower right", fontsize=8, framealpha=.85)
    return fig, frame[["method", "target", "cross_auroc_mean", "cross_unique_auroc_mean"]], "28 method-target points; both coordinates average the same three training seeds."


def figure7():
    q = pd.DataFrame(read_json(Q / "metrics_long.json"))
    native = pd.DataFrame(read_json(CONTROL / "metrics.json")).rename(columns={"source_seed": "seed"})
    rows = []
    fig, ax = plt.subplots(figsize=(6.92, 4.99))
    fig.subplots_adjust(left=.14, right=.98, bottom=.22, top=.87)
    for i, target in enumerate(reversed(DOMAINS)):
        a = q[(q.target == target) & (q.method == "q_expert_median_inductive")].set_index("seed").loc[list(SEEDS), "group_auroc"]
        controls = [("source-LR", q, "source_erm", -.12, "#3567a8"), ("matched native", native, "native78_shared_expert_median", .12, "#d89000")]
        for name, source, method, offset, color in controls:
            b = source[(source.target == target) & (source.method == method)].set_index("seed").loc[list(SEEDS), "group_auroc"]
            delta = a - b
            y = i + offset
            ax.scatter(delta, y + np.linspace(-.035, .035, len(delta)), s=22, alpha=.68, color=color, edgecolors="white", linewidths=.35, zorder=3)
            ax.errorbar(delta.mean(), y, xerr=delta.std(ddof=1), fmt="D", markersize=5, color=color, capsize=3, lw=1.4, zorder=4)
            ax.text(delta.mean(), y + (.065 if offset > 0 else -.10), f"{int((delta > 0).sum())}/9 +", color=color, fontsize=7, ha="center")
            rows.extend(dict(target=target, seed=int(seed), comparison=name, delta=float(value)) for seed, value in delta.items())
    ax.set(xlim=(-.25, .72), ylim=(-.32, 3.32), xlabel="Paired difference in target group AUROC", ylabel="Held-out target")
    ax.set_yticks(range(4), [DISPLAY[t] for t in reversed(DOMAINS)])
    ax.set_xticks(np.arange(-.2, .61, .2)); ax.grid(axis="x", color="#cccccc", lw=.8)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    handles = [Line2D([], [], marker="D", color=color, markersize=5, label="Q - " + name) for name, color in [("source-LR", "#3567a8"), ("matched native", "#d89000")]]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=2, frameon=False, fontsize=9)
    fig.text(.54, .075, "Dots: nine source-subsampling seeds; diamonds and bars: mean +/- SD.", ha="center", fontsize=8, color="#666666")
    fig.text(.54, .035, "Sign counts are descriptive over seeds.", ha="center", fontsize=8, color="#666666")
    return fig, pd.DataFrame(rows), "72 paired seed differences; eight means with sample SD (ddof=1), not confidence intervals."


def figure8():
    frame = read_csv(STAIR / "staircase_metrics.csv")
    grouped = frame.groupby(["target", "n_labeled", "stage"]).auroc_eval_group
    if not grouped.size().eq(len(SUPPORT_SEEDS)).all():
        raise ValueError("Each staircase cell must contain five support seeds")
    values = grouped.mean().unstack("stage").loc[[(t, n) for t in DOMAINS for n in (20, 50, 100)], ["S0", "S1", "S2", "S3"]]
    fig, ax = plt.subplots(figsize=(6.76, 5.66))
    fig.subplots_adjust(left=.23, right=.87, bottom=.16, top=.98)
    mesh = heatmap(ax, values.to_numpy(), values.columns, [f"{DISPLAY[t]} · N={n}" for t, n in values.index], .2)
    for y in (3, 6, 9): ax.axhline(y, color="#555555", lw=1.1)
    ax.set(xlabel="Information stage", ylabel="Target and labeled-support budget")
    cax = fig.add_axes([.895, .25, .022, .64])
    colorbar = fig.colorbar(mesh, cax=cax, label="Target group AUROC")
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    fig.text(.56, .045, "Each cell is the five-support-seed mean; color center marks chance (0.5).", ha="center", fontsize=8, color="#666666")
    return fig, values.stack().rename("group_auroc_mean").reset_index(), "48 domain-budget-stage means; five support seeds per cell. S0 repeats the same domain baseline across budgets."


def figure9():
    raw = read_csv(DEPLOY / "coordinate_rows.csv")
    frame = raw.groupby(["target", "n_labeled", "mode"], as_index=False)[["source_threshold_fpr_row", "source_threshold_tpr_row"]].mean()
    rates = frame[["source_threshold_fpr_row", "source_threshold_tpr_row"]].to_numpy()
    if len(frame) != 52 or not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError("Expected 52 finite seed-averaged operating points in [0, 1]")
    fig, ax = plt.subplots(figsize=(7.34, 5.52))
    fig.subplots_adjust(left=.10, right=.77, bottom=.15, top=.98)
    fpr = frame.source_threshold_fpr_row
    positive = fpr[fpr > 0]
    left = float(positive.min()) / 2 if len(positive) and positive.min() < 5e-5 else 5e-5
    if (fpr == 0).any():
        ax.set_xscale("symlog", linthresh=left)
        left = 0.
        xlabel = "Transferred target FPR (symlog; linear near zero)"
    else:
        ax.set_xscale("log")
        xlabel = "Transferred target FPR (log scale)"
    right = float(fpr.max()) * 1.1 if fpr.max() >= .7 else .7
    ax.set(xlim=(left, right), ylim=(-.025, 1.02), xlabel=xlabel, ylabel="Transferred target TPR")
    ax.fill_between([left, .05], .2, 1, color="#d8eadb", alpha=.5, zorder=0)
    ax.axvline(.05, color="#388744", ls="--", lw=.8); ax.axhline(.2, color="#388744", ls="--", lw=.8)
    ax.text(7e-5, .96, "FPR ≤ 0.05; TPR ≥ 0.20", color="#388744", fontsize=8)
    ax.grid(which="major", color="#cccccc", lw=.8)
    modes = [("zero_shot", "X", "zero shot"), ("full_ft", "o", "full ft"), ("target_only", "^", "target only")]
    for mode, marker, label in modes:
        for target in DOMAINS:
            rows = frame[(frame.target == target) & (frame['mode'] == mode)]
            sizes = np.full(len(rows), 25.) if mode == "zero_shot" else 24 + 12 * np.log10(rows.n_labeled.to_numpy())
            ax.scatter(rows.source_threshold_fpr_row, rows.source_threshold_tpr_row, marker=marker, s=sizes, color=COLORS[target], edgecolors="white", linewidths=.55, alpha=.78, zorder=3)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[t], markersize=4, label=DISPLAY[t]) for t in DOMAINS]
    legend = ax.legend(handles=handles, title="Target", loc="upper left", bbox_to_anchor=(1.03, 1), frameon=False, fontsize=8, title_fontsize=9)
    ax.add_artist(legend)
    handles = [Line2D([], [], marker=marker, linestyle="none", color="#666666", markersize=5, label=label) for _, marker, label in modes]
    ax.legend(handles=handles, title="Training mode", loc="upper left", bbox_to_anchor=(1.03, .63), frameon=False, fontsize=8, title_fontsize=9)
    ax.text(1.05, .34, "Larger markers:\nmore support labels", transform=ax.transAxes, fontsize=8, color="#666666")
    examples = [("bot", 10, "full_ft", "BoT N=10 full-ft", (.013, .88)), ("cic", 0, "zero_shot", "CIC zero-shot", (.045, .38)), ("unsw", 1000, "full_ft", "UNSW N=1000 full-ft", (.00015, .22))]
    for target, n, mode, label, textxy in examples:
        row = frame[(frame.target == target) & (frame.n_labeled == n) & (frame['mode'] == mode)].iloc[0]
        ax.annotate(label, (row.source_threshold_fpr_row, row.source_threshold_tpr_row), xytext=textxy, fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#888888", lw=.8), bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.3))
    return fig, frame, "732 coordinates averaged into 52 target-budget-mode points; row-level FPR/TPR at source-selected thresholds."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "RESULTS/reruns/figures")
    parser.add_argument("--figures", type=int, choices=tuple(STEMS), nargs="+", default=list(STEMS))
    parser.add_argument("--png", action="store_true", help="also export 200-dpi previews")
    parser.add_argument("--results-root", type=Path, default=ROOT / "RESULTS_PAPER")
    args = parser.parse_args()
    from CODE.analysis import paper_results as results
    results.configure_results(args.results_root)
    globals().update({name: getattr(results, name) for name in ("RQ1", "Q", "CONTROL", "STAIR", "DEPLOY")})
    try: out = output_directory(args.output_dir)
    except ValueError as error: parser.error(str(error))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10, "axes.edgecolor": "#cccccc", "axes.linewidth": .8, "xtick.color": "#262626", "ytick.color": "#262626", "text.color": "#262626", "axes.labelcolor": "#262626", "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "nids-paper-figures"})
    manifest = {"generator": "CODE/figures/plot_empirical_figures.py", "figures": []}
    for number in dict.fromkeys(args.figures):
        INPUTS.clear()
        fig, values, interpretation = globals()[f"figure{number}"]()
        stem = STEMS[number]
        outputs = []
        for extension in (["pdf", "svg", "png"] if args.png else ["pdf", "svg"]):
            path = out / f"{stem}.{extension}"
            fig.savefig(path, dpi=200)
            outputs.append(path)
        path = out / f"{stem}.csv"; values.to_csv(path, index=False); outputs.append(path)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["figures"].append(dict(number=number, aggregation=interpretation,
            inputs=[dict(path=results.input_path(p), sha256=sha(p)) for p in sorted(INPUTS)],
            outputs=[dict(path=p.name, sha256=sha(p)) for p in outputs]))
        plt.close(fig)
        print(f"Rendered {stem}: {len(values)} plotted records")
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
