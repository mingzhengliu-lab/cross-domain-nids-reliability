"""CPU control separating the Q representation from source-expert structure.

No target labels are loaded until every candidate prediction for a cell is
persisted. These are the manuscript controls on previously inspected domains.
"""
from __future__ import annotations

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
DATA = ROOT / "DATA/nfv2_processed"
DEFAULT_OUT = ROOT / "RESULTS/reruns/matched_control"
sys.path.insert(0, str(ROOT / "CODE/common"))
from nidsbed.data_utils import (apply_source_train_preprocessor,
                              fit_source_train_preprocessor,
                              load_domain_split)

DOMAINS = ("unsw", "cic", "bot", "ton")
DEPS = (Path(__file__).resolve(), ROOT / "CODE/common/nidsbed/data_utils.py")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in DEPS}


def zi_flags(x):
    x = np.asarray(x, dtype=np.float64)
    bad = (~np.isfinite(x)) | (np.abs(x) > 1e8)
    zero = (~bad) & (x == 0)
    return zero, bad


def fit_zi_edges(x, n_bins=129, cap=60000):
    # Exact legacy Q v1 transform, frozen here so concurrent metadata changes
    # in the formal runner cannot alter this control during execution.
    x = np.asarray(x, dtype=np.float64)
    if len(x) > cap:
        rng = np.random.default_rng(20260912)
        x = x[rng.choice(len(x), cap, replace=False)]
    edges = np.zeros((x.shape[1], n_bins - 2), dtype=np.float64)
    for j in range(x.shape[1]):
        v = x[:, j]
        bad = (~np.isfinite(v)) | (np.abs(v) > 1e8)
        use = (~bad) & (v != 0)
        if use.any():
            q = np.quantile(v[use], np.linspace(0, 1, n_bins), method="linear")
            edges[j] = np.maximum.accumulate(q[1:-1])
    return edges


def transform_zi(x, edges):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((len(x), 3 * x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        v = x[:, j]
        bad = (~np.isfinite(v)) | (np.abs(v) > 1e8)
        zero = (~bad) & (v == 0)
        vv = np.where((~bad) & (~zero), v, 0.0)
        rank = np.searchsorted(edges[j], vv, side="right") / (len(edges[j]) + 1)
        rank = rank * 2 - 1
        rank[bad | zero] = 0
        out[:, j] = rank
        out[:, x.shape[1] + j] = zero
        out[:, 2 * x.shape[1] + j] = bad
    return out


def fit_native(x):
    prep = fit_source_train_preprocessor(x)
    clean, _ = apply_source_train_preprocessor(x, prep)
    return prep, StandardScaler().fit(clean)


def native_views(x, prep, scaler):
    clean, native_bad = apply_source_train_preprocessor(x, prep)
    values = np.clip(scaler.transform(clean), -8, 8)
    zero, q_bad = zi_flags(x)
    return {
        "native39": values,
        "native78": np.c_[values, native_bad],
        # Identical zero/invalid channels to Q; the native numeric channel
        # retains source-fitted native imputation/capping/scaling.
        "native117": np.c_[values, zero, q_bad].astype(np.float32),
    }


def fit_predict(train_x, train_y, test_x):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=1, solver="liblinear", class_weight="balanced",
                                 max_iter=160, random_state=0).fit(train_x, train_y)
    return clf.decision_function(test_x), {
        "n_iter": int(clf.n_iter_.max()),
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "feature_count": int(train_x.shape[1]),
    }


def score_metrics(y, score, gid):
    ug, inv = np.unique(gid, return_inverse=True)
    count = np.bincount(inv)
    gy = np.bincount(inv, weights=y) / count
    if np.any((gy != 0) & (gy != 1)):
        raise ValueError("Mixed-label duplicate group; no silent majority vote")
    gs = np.bincount(inv, weights=score) / count
    def one(yy, ss):
        fpr, tpr, _ = roc_curve(yy, ss)
        return {"auroc": float(roc_auc_score(yy, ss)),
                "auprc": float(average_precision_score(yy, ss)),
                "oracle_tpr_at_fpr1": float(tpr[fpr <= .01].max())}
    return {**{f"row_{k}": v for k, v in one(y, score).items()},
            **{f"group_{k}": v for k, v in one(gy, gs).items()},
            "n_rows": len(y), "n_groups": len(ug), "mixed_group_count": 0}


def run_cell(target, source_seed, max_rows, output):
    started = time.time()
    start_hashes = hashes()
    cell_dir = Path(output) / f"{target}_seed{source_seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    source_domains = [d for d in DOMAINS if d != target]
    splits = {d: load_domain_split(DATA, d) for d in DOMAINS}
    rng = np.random.default_rng(source_seed + DOMAINS.index(target))
    parts, source_indices = [], {}
    for d in source_domains:
        train_idx = splits[d]["train_idx"]
        idx = train_idx[rng.choice(len(train_idx), min(max_rows, len(train_idx)), replace=False)]
        source_indices[d] = idx
        xx = np.load(DATA / f"{d}_X.npy", mmap_mode="r")[idx].astype(np.float32)
        yy = np.load(DATA / f"{d}_y.npy", mmap_mode="r")[idx].astype(np.int8)
        parts.append((d, xx, yy))
    target_x = np.load(DATA / f"{target}_X.npy", mmap_mode="r")
    test_idx = splits[target]["test_idx"]
    tx = np.asarray(target_x[test_idx], dtype=np.float32)
    test_gid = splits[target]["row_group_id"][test_idx]
    sx = np.concatenate([x for _, x, _ in parts])
    sy = np.concatenate([y for _, _, y in parts])
    scores, fit_audit = {}, {}
    with threadpool_limits(limits=1):
        pooled_prep, pooled_scaler = fit_native(sx)
        pooled_s = native_views(sx, pooled_prep, pooled_scaler)
        pooled_t = native_views(tx, pooled_prep, pooled_scaler)
        scores["native78_pooled"], fit_audit["native78_pooled"] = fit_predict(
            pooled_s["native78"], sy, pooled_t["native78"])
        target_edges = fit_zi_edges(target_x[splits[target]["train_idx"]])
        tq = transform_zi(tx, target_edges)
        expert_predictions = {name: [] for name in
                              ("native39", "native78", "native117", "native78_shared", "q117")}
        for d, xx, yy in parts:
            prep, scaler = fit_native(xx)
            sv = native_views(xx, prep, scaler)
            tv = native_views(tx, prep, scaler)
            sv["native78_shared"] = native_views(xx, pooled_prep, pooled_scaler)["native78"]
            tv["native78_shared"] = pooled_t["native78"]
            sv["q117"] = transform_zi(xx, fit_zi_edges(xx))
            tv["q117"] = tq
            for name in expert_predictions:
                pred, audit = fit_predict(sv[name], yy, tv[name])
                fit_audit[f"{name}_{d}"] = audit
                expert_predictions[name].append(pred)
        for name, predictions in expert_predictions.items():
            ep = np.asarray(predictions)
            scores[f"{name}_expert_mean"] = ep.mean(axis=0)
            scores[f"{name}_expert_median"] = np.median(ep, axis=0)
    # The complete score artifact is physically persisted before target-label
    # access. No candidate or hyperparameter selection occurs in this runner.
    np.savez_compressed(cell_dir / "frozen_predictions.npz", test_idx=test_idx,
                        test_gid=test_gid, **scores)
    np.savez_compressed(cell_dir / "source_indices.npz", **source_indices)
    freeze_hash = file_hash(cell_dir / "frozen_predictions.npz")
    freeze = {"target": target, "source_seed": source_seed,
              "target_labels_loaded": False, "prediction_sha256": freeze_hash,
              "dependencies_start": start_hashes, "fit_audit": fit_audit,
              "source_indices_sha256": file_hash(cell_dir / "source_indices.npz")}
    (cell_dir / "prediction_freeze.json").write_text(json.dumps(freeze, indent=2))
    # Evaluation is the first access to this target's label array in this cell.
    target_y = np.load(DATA / f"{target}_y.npy", mmap_mode="r")
    ty = np.asarray(target_y[test_idx], dtype=np.int8)
    rows = [{"target": target, "source_seed": source_seed, "method": name,
             "source_rows_per_domain": max_rows, **score_metrics(ty, s, test_gid)}
            for name, s in scores.items()]
    assert freeze_hash == file_hash(cell_dir / "frozen_predictions.npz")
    end_hashes = hashes()
    if start_hashes != end_hashes:
        raise RuntimeError("Code dependency changed while cell was running")
    (cell_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    (cell_dir / "run_audit.json").write_text(json.dumps({
        "dependencies_end": end_hashes, "dependency_hashes_match": True,
        "prediction_freeze_sha256": file_hash(cell_dir / "prediction_freeze.json"),
        "elapsed_seconds": time.time() - started}, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--targets", default=",".join(DOMAINS))
    ap.add_argument("--source-seeds", default="1,7,42,99,1234,2026,2027,2028,55555")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    targets = args.targets.split(",")
    if not set(targets) <= set(DOMAINS):
        ap.error("unknown target")
    seeds = [int(s) for s in args.source_seeds.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "matched_source_experts_exploratory_v1", "targets": targets,
                "source_seeds": seeds, "source_rows_per_domain": args.max_rows,
                "model": "balanced LR C1 liblinear max_iter160 random_state0",
                "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                "dependencies_start": hashes(),
                "target_feature_access": "Only Q fits target-train unlabeled CDF; raw experts source-only",
                "target_label_access": "All cell scores persisted before cell target-label evaluation",
                "native117_channels": "native source-imputed capped scaled values + exact Q zero/bad flags",
                "native117_numeric_invalid_policy": "native >=1e9; Q numeric >1e8; explicitly distinct representations",
                "estimand": "one mean score per homogeneous exact duplicate group",
                "selection": "none; report every fixed candidate",
                "warning": "Previously inspected domains; exploratory, no independent confirmation or deployment safety claim"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_cell, t, s, args.max_rows, args.output): (t, s)
                for s in seeds for t in targets}
        for future in as_completed(jobs):
            result = future.result()
            rows.extend(result)
            rows.sort(key=lambda x: (x["source_seed"], DOMAINS.index(x["target"]), x["method"]))
            (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"cell": jobs[future], "completed_cells": len(rows) // len(result),
                              "group_auroc": {r["method"]: r["group_auroc"] for r in result}}), flush=True)
    manifest.update({"dependencies_end": hashes(), "rows": len(rows), "cells": len(targets) * len(seeds)})
    if manifest["dependencies_start"] != manifest["dependencies_end"]:
        raise RuntimeError("Code dependency changed during run")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
