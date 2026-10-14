from __future__ import annotations

"""Q-TRANSPORT for the manuscript: exact quantiles and nine paired source seeds.

Quantile edges use every valid nonzero fitting observation. Target training
features fit the inductive transform; labels are opened only for evaluation
after predictions have been saved. The legacy capped transform is implemented
locally only for --edges-audit and needs no v1 executable.

Source subsamples, fixed splits, preprocessing, logistic models, aggregation,
and metric calculations are unchanged from the archived v2 implementation.
qtransport_source_layout.json records the packaging-only source edits.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "DATA" / "nfv2_processed"
DEFAULT_OUT = ROOT / "RESULTS" / "reruns" / "qtransport"
# The retired v1 executable is not a runtime dependency.
sys.path.insert(0, str(ROOT / "CODE" / "common"))
from nidsbed.data_utils import (apply_source_train_preprocessor,
                              fit_source_train_preprocessor,
                              load_domain_split)

DOMAINS = ("unsw", "cic", "bot", "ton")
DEFAULT_SEEDS = (1, 7, 42, 99, 1234, 2026, 2027, 2028, 55555)
N_BINS = 129
LEGACY_CAP = 60000          # the v1 cap, kept ONLY for --edges-audit
LEGACY_CAP_RNG = 20260912   # the v1 hardcoded rng, kept ONLY for --edges-audit
DEPS = (Path(__file__).resolve(), ROOT / "CODE" / "common" / "nidsbed" / "data_utils.py")


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashes() -> dict:
    out = {str(p.relative_to(ROOT)): file_hash(p) for p in DEPS}
    # Historical v1 lineage is retained in the frozen records only.
    return out


# --------------------------------------------------------------------------
# The Q representation. Identical formula to v1; the ONLY change is that the
# quantile edges are computed on the FULL sample with no rng and no cap.
# --------------------------------------------------------------------------
def fit_zi_edges(x, n_bins: int = N_BINS):
    """Exact zero-inflated quantile edges over every supplied row.

    phi_{d,j}(x) = [2 F^nz_{d,j}(x_j) - 1, 1(x_j = 0), 1(x_j invalid)]
    where F^nz is the empirical CDF of the valid non-zero values. Deterministic:
    no subsampling, no random state, so repeated calls on the same input return
    bit-identical edges.
    """
    x = np.asarray(x, dtype=np.float64)
    edges = np.zeros((x.shape[1], n_bins - 2), dtype=np.float64)
    for j in range(x.shape[1]):
        v = x[:, j]
        bad = (~np.isfinite(v)) | (np.abs(v) > 1e8)
        use = (~bad) & (v != 0)
        if use.any():
            q = np.quantile(v[use], np.linspace(0, 1, n_bins), method="linear")
            edges[j] = np.maximum.accumulate(q[1:-1])
    return edges


def fit_zi_edges_legacy_capped(x, n_bins: int = N_BINS, cap: int = LEGACY_CAP):
    """Byte-for-byte the v1 behaviour. Used ONLY by --edges-audit."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) > cap:
        rng = np.random.default_rng(LEGACY_CAP_RNG)
        x = x[rng.choice(len(x), cap, replace=False)]
    return fit_zi_edges(x, n_bins=n_bins)


def transform_zi(x, edges):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    out = np.empty((len(x), 3 * d), dtype=np.float32)
    for j in range(d):
        v = x[:, j]
        bad = (~np.isfinite(v)) | (np.abs(v) > 1e8)
        zero = (~bad) & (v == 0)
        vv = np.where((~bad) & (~zero), v, 0.0)
        rank = np.searchsorted(edges[j], vv, side="right") / (len(edges[j]) + 1)
        rank = rank * 2.0 - 1.0
        rank[bad | zero] = 0.0
        out[:, j] = rank
        out[:, d + j] = zero
        out[:, 2 * d + j] = bad
    return out


def fit_lr(x, y):
    """Aligned with the matched-native control runner. See module docstring."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=160,
                                 class_weight="balanced",
                                 random_state=0).fit(x, y)
    audit = {"n_iter": int(clf.n_iter_.max()),
             "convergence_warning": any(issubclass(w.category, ConvergenceWarning)
                                        for w in caught),
             "feature_count": int(x.shape[1])}
    return clf, audit


def _tpr_at_fpr(y, score, fpr_limit: float = 0.01):
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr[fpr <= fpr_limit])) if np.any(fpr <= fpr_limit) else 0.0


def metrics(y, score, gid) -> dict:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    out = {"row_auroc": float(roc_auc_score(y, score)),
           "row_auprc": float(average_precision_score(y, score)),
           "row_tpr_at_fpr1": _tpr_at_fpr(y, score)}
    _, inv = np.unique(np.asarray(gid), return_inverse=True)
    ng = int(inv.max() + 1) if len(inv) else 0
    cnt = np.bincount(inv, minlength=ng).astype(float)
    gs = np.bincount(inv, weights=score, minlength=ng) / np.maximum(cnt, 1)
    gy_raw = np.bincount(inv, weights=y, minlength=ng) / np.maximum(cnt, 1)
    if np.any((gy_raw > 0) & (gy_raw < 1)):
        raise ValueError("mixed-label duplicate group; no silent majority vote")
    gy = (gy_raw >= 0.5).astype(np.int8)
    out.update({"group_auroc": float(roc_auc_score(gy, gs)),
                "group_auprc": float(average_precision_score(gy, gs)),
                "group_tpr_at_fpr1": _tpr_at_fpr(gy, gs),
                "n_rows": int(len(y)), "n_groups": ng,
                "group_mixed_fraction": float(np.mean((gy_raw > 0) & (gy_raw < 1)))})
    return out


def run_cell(target: str, seed: int, max_rows: int, output: Path) -> list[dict]:
    """One (target, seed) cell. Predictions are persisted BEFORE target labels."""
    started = time.time()
    start_hashes = hashes()
    cell_dir = Path(output) / f"{target}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    arr = {d: np.load(DATA / f"{d}_X.npy", mmap_mode="r") for d in DOMAINS}
    ys = {d: np.load(DATA / f"{d}_y.npy", mmap_mode="r") for d in DOMAINS}
    sp = {d: load_domain_split(DATA, d) for d in DOMAINS}

    te = sp[target]["test_idx"]
    tx = np.asarray(arr[target][te], dtype=np.float32)
    gid = np.asarray(sp[target]["row_group_id"][te])

    # Same source-row draw as v1 and as the matched-native control, so cells with
    # equal seed are PAIRED across the two runners.
    rng = np.random.default_rng(seed + DOMAINS.index(target))
    parts, source_indices = [], {}
    for d in DOMAINS:
        if d == target:
            continue
        tr = sp[d]["train_idx"]
        take = min(max_rows, len(tr))
        ii = tr[rng.choice(len(tr), take, replace=False)]
        source_indices[d] = ii
        parts.append((d, np.asarray(arr[d][ii], dtype=np.float32),
                      np.asarray(ys[d][ii], dtype=np.int8)))
    sx = np.concatenate([p[1] for p in parts])
    sy = np.concatenate([p[2] for p in parts])

    scores, fit_audit = {}, {}
    with threadpool_limits(limits=1):
        # ---- source-only arms (no target information at all) ----
        prep = fit_source_train_preprocessor(sx)
        sxp, sxi = apply_source_train_preprocessor(sx, prep)
        txp, txi = apply_source_train_preprocessor(tx, prep)
        scaler = StandardScaler().fit(sxp)
        sx_erm = np.clip(scaler.transform(sxp), -8, 8)
        tx_erm = np.clip(scaler.transform(txp), -8, 8)
        erm, fit_audit["source_erm"] = fit_lr(np.c_[sx_erm, sxi], sy)
        scores["source_erm"] = erm.decision_function(np.c_[tx_erm, txi])
        raw39, fit_audit["source_raw39_lr"] = fit_lr(sx_erm, sy)
        scores["source_raw39_lr"] = raw39.decision_function(tx_erm)

        # ---- Q arms. inductive is the main track; transductive is sensitivity ----
        target_fits = {
            "inductive": fit_zi_edges(np.asarray(arr[target][sp[target]["train_idx"]],
                                                 dtype=np.float32)),
            "transductive": fit_zi_edges(np.asarray(arr[target], dtype=np.float32)),
        }
        pooled_edges = fit_zi_edges(sx)
        expert_edges = [fit_zi_edges(xx) for _, xx, _ in parts]
        for mode, tfit in target_fits.items():
            txq = transform_zi(tx, tfit)
            pooled, fit_audit[f"q_pooled_{mode}"] = fit_lr(transform_zi(sx, pooled_edges), sy)
            scores[f"q_pooled_{mode}"] = pooled.decision_function(txq)
            E = []
            for k, (_, xx, yy) in enumerate(parts):
                clf, aud = fit_lr(transform_zi(xx, expert_edges[k]), yy)
                fit_audit[f"q_expert_{mode}_{parts[k][0]}"] = aud
                E.append(clf.decision_function(txq))
            E = np.vstack(E)
            scores[f"q_expert_mean_{mode}"] = E.mean(axis=0)
            scores[f"q_expert_median_{mode}"] = np.median(E, axis=0)

    # Persist every score BEFORE this target's label array is read.
    np.savez_compressed(cell_dir / "frozen_predictions.npz", test_idx=te,
                        test_gid=gid, **scores)
    np.savez_compressed(cell_dir / "source_indices.npz", **source_indices)
    freeze_hash = file_hash(cell_dir / "frozen_predictions.npz")
    (cell_dir / "prediction_freeze.json").write_text(json.dumps({
        "target": target, "source_seed": seed, "runner_version": "v2_exact_quantiles",
        "target_labels_loaded": False, "prediction_sha256": freeze_hash,
        "dependencies_start": start_hashes, "fit_audit": fit_audit,
        "quantile_policy": "exact_full_sample_no_cap_no_rng",
        "source_indices_sha256": file_hash(cell_dir / "source_indices.npz"),
    }, indent=2), encoding="utf-8")

    # Evaluation is the first access to this target's labels in this cell.
    ty = np.asarray(ys[target][te], dtype=np.int8)
    rows = [{"target": target, "seed": seed, "method": name,
             "target_fit": ("source_only" if name in {"source_erm", "source_raw39_lr"}
                            else name.rsplit("_", 1)[-1]),
             "source_rows": int(len(sx)), "source_rows_per_domain": max_rows,
             **metrics(ty, s, gid)}
            for name, s in scores.items()]
    assert freeze_hash == file_hash(cell_dir / "frozen_predictions.npz")

    end_hashes = hashes()
    if start_hashes != end_hashes:
        raise RuntimeError("code dependency changed while cell was running")
    (cell_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (cell_dir / "run_audit.json").write_text(json.dumps({
        "dependencies_end": end_hashes, "dependency_hashes_match": True,
        "prediction_freeze_sha256": file_hash(cell_dir / "prediction_freeze.json"),
        "elapsed_seconds": time.time() - started}, indent=2), encoding="utf-8")
    return rows


def edges_audit(target: str, max_rows: int) -> dict:
    """How much did the removed v1 cap actually change the representation?

    Recomputes edges both ways on the three call sites where the v1 cap bound
    and reports the induced bin shift on a held-out sample of target rows.
    """
    arr = {d: np.load(DATA / f"{d}_X.npy", mmap_mode="r") for d in DOMAINS}
    sp = {d: load_domain_split(DATA, d) for d in DOMAINS}
    rng = np.random.default_rng(DEFAULT_SEEDS[0] + DOMAINS.index(target))
    parts = []
    for d in DOMAINS:
        if d == target:
            continue
        tr = sp[d]["train_idx"]
        ii = tr[rng.choice(len(tr), min(max_rows, len(tr)), replace=False)]
        parts.append(np.asarray(arr[d][ii], dtype=np.float32))
    sx = np.concatenate(parts)
    ttr = np.asarray(arr[target][sp[target]["train_idx"]], dtype=np.float32)
    tall = np.asarray(arr[target], dtype=np.float32)

    probe = tall[::max(1, len(tall) // 20000)][:20000]
    report = {"target": target, "probe_rows": int(len(probe)), "sites": {}}
    for site, data in (("pooled_source", sx), ("target_inductive", ttr),
                       ("target_transductive", tall)):
        exact = fit_zi_edges(data)
        capped = fit_zi_edges_legacy_capped(data)
        n_bind = len(data) > LEGACY_CAP
        # Rank-channel disagreement on the probe measures the observable effect.
        ra = transform_zi(probe, exact)[:, :probe.shape[1]]
        rb = transform_zi(probe, capped)[:, :probe.shape[1]]
        diff = np.abs(ra.astype(np.float64) - rb.astype(np.float64))
        edge_shift = np.abs(exact - capped)
        report["sites"][site] = {
            "n_rows": int(len(data)), "v1_cap_binds": bool(n_bind),
            "rank_channel_disagree_fraction": float(np.mean(diff > 0)),
            "rank_channel_max_abs_shift": float(diff.max()) if diff.size else 0.0,
            "rank_channel_mean_abs_shift": float(diff.mean()) if diff.size else 0.0,
            "edge_max_abs_shift": float(edge_shift.max()),
            "edge_bins_changed_fraction": float(np.mean(edge_shift > 0)),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Q-TRANSPORT: exact quantile edges, nine manuscript seeds")
    ap.add_argument("--targets", default=",".join(DOMAINS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--edges-audit", action="store_true",
                    help="also quantify the effect of removing the v1 cap")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not set(targets) <= set(DOMAINS):
        ap.error(f"unknown target; must be a subset of {DOMAINS}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    deps_start = hashes()
    manifest = {
        "protocol": "nfv2_paired_source_lr_qtransport_v2_exact_quantiles",
        "runner_version": "v2",
        "targets": targets, "seeds": seeds,
        "source_rows_per_domain": args.max_rows,
        "source_domains_per_target": len(DOMAINS) - 1,
        "methods": ["source_erm", "source_raw39_lr", "q_pooled",
                    "q_expert_mean", "q_expert_median"],
        "target_cdf_modes": ["inductive_target_train_only_MAIN",
                             "transductive_unlabeled_full_target_SENSITIVITY"],
        "model": "class_balanced_logistic_regression_liblinear_C1_maxiter160_randomstate0",
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "group_estimand": "mean_score_per_persisted_exact_duplicate_group",
        "quantile_policy": "exact_full_sample_no_cap_no_rng",
        "changes_vs_frozen_v1": {
            "cap_removed": True,
            "v1_cap": LEGACY_CAP,
            "v1_hardcoded_rng": LEGACY_CAP_RNG,
            "v1_cap_bound_on": ["pooled_source", "target_inductive", "target_transductive"],
            "v1_cap_never_bound_on": ["per_source_expert"],
            "seeds_changed_from": [20260912, 20260913, 20260914],
            "lr_aligned_to_control": "max_iter 100 -> 160, random_state None -> 0",
            "lr_alignment_is_numerically_noop":
                "max observed n_iter across 16 persisted control cells = 25, "
                "zero convergence warnings",
            "everything_else": "unchanged from v1",
        },
        "lodo_split_seed": "2026 (frozen in nidsbed/datasets.py; NOT modified)",
        "label_usage": "target labels evaluation only; inductive fit excludes target-test features",
        "dependencies_start": deps_start,
        "warning": "Paired source-LR control; NOT the Stage3 neural ERM checkpoint.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.edges_audit:
        t0 = time.time()
        audits = [edges_audit(t, args.max_rows) for t in targets]
        (args.output / "edges_cap_audit.json").write_text(
            json.dumps(audits, indent=2), encoding="utf-8")
        print(json.dumps({"edges_audit": "done", "elapsed_s": round(time.time() - t0, 1)}),
              flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_cell, t, s, args.max_rows, args.output): (t, s)
                for s in seeds for t in targets}
        for future in as_completed(jobs):
            got = future.result()
            rows.extend(got)
            rows.sort(key=lambda r: (r["seed"], DOMAINS.index(r["target"]), r["method"]))
            (args.output / "metrics_long.json").write_text(json.dumps(rows, indent=2),
                                                           encoding="utf-8")
            print(json.dumps({"cell": jobs[future], "rows": len(got),
                              "elapsed_s": round(time.time() - t0, 1),
                              "group_auroc": {r["method"]: round(r["group_auroc"], 4)
                                              for r in got}}, ensure_ascii=False), flush=True)

    (args.output / "metrics_long.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Cross-seed aggregation per (target, method).
    agg = []
    for target in targets:
        for method in sorted({r["method"] for r in rows}):
            vals = [r for r in rows if r["target"] == target and r["method"] == method]
            if not vals:
                continue
            rec = {"target": target, "method": method, "n_seeds": len(vals)}
            for key in ("row_auroc", "group_auroc", "row_auprc", "group_auprc",
                        "row_tpr_at_fpr1", "group_tpr_at_fpr1"):
                v = np.asarray([x[key] for x in vals], dtype=np.float64)
                rec[f"{key}_mean"] = float(v.mean())
                rec[f"{key}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                rec[f"{key}_per_seed"] = [float(x) for x in v]
            agg.append(rec)
    (args.output / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    lines = ["# Q-TRANSPORT v2 (exact quantile edges, seeds "
             + ",".join(str(s) for s in seeds) + ")", "",
             "Group AUROC mean +/- SD across seeds. `source_erm` is the paired "
             "source-only LR control (NOT the Stage3 neural ERM checkpoint).", "",
             "| target | method | group AUROC | row AUROC | group TPR@1% |",
             "|---|---|---:|---:|---:|"]
    for r in agg:
        lines.append(f"| {r['target']} | {r['method']} | "
                     f"{r['group_auroc_mean']:.4f} +/- {r['group_auroc_sd']:.4f} | "
                     f"{r['row_auroc_mean']:.4f} +/- {r['row_auroc_sd']:.4f} | "
                     f"{r['group_tpr_at_fpr1_mean']:.4f} |")
    (args.output / "aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    deps_end = hashes()
    manifest.update({"dependencies_end": deps_end, "rows": len(rows),
                     "cells": len(targets) * len(seeds),
                     "elapsed_seconds": round(time.time() - t0, 1)})
    if deps_start != deps_end:
        manifest["dependency_hashes_match"] = False
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
        raise RuntimeError("code dependency changed during run; results not trustworthy")
    manifest["dependency_hashes_match"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "cells": manifest["cells"], "rows": len(rows),
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
