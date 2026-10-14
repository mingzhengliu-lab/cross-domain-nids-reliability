#!/usr/bin/env python3
"""
CPU tabular baselines for the LODO NIDS benchmark.

The script trains four strong tabular baselines under the same source-only
LODO split used by the neural DG benchmark:

  * Logistic Regression
  * HistGradientBoosting
  * ExtraTrees
  * LightGBM

For each target domain and method it runs a small source-validation hyperparameter
search, selects the best source-validation metric (default val_auroc_macro),
re-fits three final seeds, saves target scores, and writes aggregate tables plus
prior/operational diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
for import_root in (COMMON_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

METHODS = ["logreg", "histgb", "extratrees", "lightgbm"]
TARGETS = ["unsw", "cic", "bot", "ton"]
DEFAULT_SEEDS = [2026, 2027, 2028]
DEFAULT_SEARCH_SEEDS = [42, 123]
PRIORS = [0.05, 0.20, 0.50, 0.80]
FPR_POINTS = [0.01, 0.001, 0.0001]

METHOD_DISPLAY = {
    "logreg": "LogReg",
    "histgb": "HistGB",
    "extratrees": "ExtraTrees",
    "lightgbm": "LightGBM",
}
TARGET_DISPLAY = {
    "unsw": "UNSW",
    "cic": "CIC",
    "bot": "BoT",
    "ton": "ToN",
}


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(p) for p in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_list(value: str, allowed: list[str]) -> list[str]:
    if value.lower() == "all":
        return list(allowed)
    out = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit(f"Invalid values {bad}; allowed values are {allowed}")
    return out


def parse_seeds(value: str, default: list[int] | None = None) -> list[int]:
    if default is None:
        default = DEFAULT_SEEDS
    if value.lower() == "default":
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_lodo_arrays(data_dir: str, target: str) -> dict[str, Any]:
    from nidsbed.datasets import LODODataLoader
    loader = LODODataLoader(data_dir=data_dir, target_domain=target)
    loader.prepare()

    train_sets = loader.get_train_datasets()
    X_train = np.concatenate([ds.X.numpy() for ds in train_sets], axis=0)
    y_train = np.concatenate([ds.y.numpy() for ds in train_sets], axis=0).astype(np.int64)

    val_per_domain = loader.get_source_val_per_domain()
    X_val = np.concatenate([v[0] for v in val_per_domain.values()], axis=0)
    y_val = np.concatenate([v[1] for v in val_per_domain.values()], axis=0).astype(np.int64)

    X_in, y_in = loader.get_in_domain_eval()
    X_cross, y_cross = loader.get_cross_domain_eval()

    return {
        "target": target,
        "sources": loader.source_domains,
        "n_features": loader.n_features,
        "X_train": X_train.astype(np.float32, copy=False),
        "y_train": y_train,
        "val_per_domain": val_per_domain,
        "X_val": X_val.astype(np.float32, copy=False),
        "y_val": y_val,
        "X_in": X_in.astype(np.float32, copy=False),
        "y_in": y_in.astype(np.int64, copy=False),
        "X_cross": X_cross.astype(np.float32, copy=False),
        "y_cross": y_cross.astype(np.int64, copy=False),
    }


def hp_grid(method: str) -> list[dict[str, Any]]:
    if method == "logreg":
        return [
            {"C": C, "class_weight": cw, "max_iter": 250}
            for C, cw in itertools.product([0.1, 1.0, 10.0], ["balanced", None])
        ]
    if method == "histgb":
        return [
            {
                "learning_rate": lr,
                "max_iter": it,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": 40,
                "l2_regularization": l2,
            }
            for lr, it, leaves, l2 in [
                (0.05, 200, 31, 0.0),
                (0.10, 200, 31, 0.0),
                (0.05, 300, 63, 0.0),
                (0.10, 300, 63, 1e-3),
                (0.03, 400, 63, 1e-2),
                (0.10, 200, 127, 1e-3),
            ]
        ]
    if method == "extratrees":
        return [
            {
                "n_estimators": n,
                "max_features": mf,
                "min_samples_leaf": leaf,
                "class_weight": "balanced_subsample",
            }
            for n, mf, leaf in [
                (300, "sqrt", 1),
                (500, "sqrt", 1),
                (300, 0.70, 5),
                (500, 0.70, 5),
                (300, 1.00, 10),
                (500, 1.00, 10),
            ]
        ]
    if method == "lightgbm":
        return [
            {
                "n_estimators": n,
                "learning_rate": lr,
                "num_leaves": leaves,
                "min_child_samples": child,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_lambda": reg,
            }
            for n, lr, leaves, child, reg in [
                (500, 0.05, 31, 40, 0.0),
                (700, 0.03, 63, 40, 0.0),
                (500, 0.10, 31, 80, 1e-3),
                (700, 0.05, 63, 80, 1e-2),
                (900, 0.03, 127, 100, 1e-2),
                (500, 0.10, 63, 100, 1e-2),
            ]
        ]
    raise ValueError(method)


def select_hp_configs(method: str, n_search: int, target: str) -> list[tuple[int, dict[str, Any]]]:
    grid = hp_grid(method)
    indexed = list(enumerate(grid))
    if n_search >= len(indexed):
        return indexed
    rng = np.random.RandomState(stable_seed("tabular_hp", method, target))
    idx = sorted(rng.choice(len(indexed), size=n_search, replace=False).tolist())
    return [indexed[i] for i in idx]


def make_estimator(method: str, hp: dict[str, Any], seed: int, threads: int):
    if method == "logreg":
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=float(hp["C"]),
            class_weight=hp["class_weight"],
            max_iter=int(hp.get("max_iter", 250)),
            n_jobs=threads,
            random_state=seed,
        )

    if method == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=float(hp["learning_rate"]),
            max_iter=int(hp["max_iter"]),
            max_leaf_nodes=int(hp["max_leaf_nodes"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            l2_regularization=float(hp["l2_regularization"]),
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=25,
            random_state=seed,
        )

    if method == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=int(hp["n_estimators"]),
            max_features=hp["max_features"],
            min_samples_leaf=int(hp["min_samples_leaf"]),
            class_weight=hp["class_weight"],
            bootstrap=False,
            n_jobs=threads,
            random_state=seed,
        )

    if method == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError(
                "LightGBM is not installed. Run `pip install -r requirements.txt` "
                "or exclude it with --methods logreg,histgb,extratrees."
            ) from exc

        return LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=int(hp["n_estimators"]),
            learning_rate=float(hp["learning_rate"]),
            num_leaves=int(hp["num_leaves"]),
            min_child_samples=int(hp["min_child_samples"]),
            subsample=float(hp["subsample"]),
            colsample_bytree=float(hp["colsample_bytree"]),
            reg_lambda=float(hp["reg_lambda"]),
            class_weight="balanced",
            n_jobs=threads,
            random_state=seed,
            force_col_wise=True,
            verbosity=-1,
        )

    raise ValueError(method)


def fit_estimator(estimator, method: str, hp: dict[str, Any], seed: int, data: dict[str, Any]):
    X_train = data["X_train"]
    y_train = data["y_train"]

    if method == "histgb":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        estimator.fit(X_train, y_train, sample_weight=weights)
        return estimator

    if method == "lightgbm":
        try:
            from lightgbm import early_stopping, log_evaluation
            estimator.fit(
                X_train,
                y_train,
                eval_set=[(data["X_val"], data["y_val"])],
                eval_metric="auc",
                callbacks=[early_stopping(40, verbose=False), log_evaluation(period=0)],
            )
        except TypeError:
            estimator.fit(X_train, y_train)
        return estimator

    estimator.fit(X_train, y_train)
    return estimator


def predict_scores(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
    else:
        scores = estimator.predict(X).astype(np.float32)
    return np.asarray(scores, dtype=np.float32)


def metrics_from_scores(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = (scores >= 0.5).astype(np.int64)
    try:
        auroc = float(roc_auc_score(y, scores))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, scores))
    except ValueError:
        auprc = float("nan")

    return {
        "attack_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "attack_ratio": float(np.mean(y == 1)),
        "n_samples": int(len(y)),
    }


def source_val_macro(estimator, val_per_domain: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    per_domain = {}
    per_domain_full = {}
    for domain, (X, y) in val_per_domain.items():
        scores = predict_scores(estimator, X)
        m = metrics_from_scores(y, scores)
        per_domain[domain] = m["attack_f1"]
        per_domain_full[domain] = m

    def _macro(key: str) -> float:
        vals = [m.get(key, float("nan")) for m in per_domain_full.values()]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "val_attack_f1_macro": float(np.mean(list(per_domain.values()))),
        "val_auroc_macro": _macro("auroc"),
        "val_auprc_macro": _macro("auprc"),
        "val_attack_f1_per_domain": per_domain,
        "val_metrics_per_domain": per_domain_full,
    }


def run_one(method: str, target: str, hp_id: int, hp: dict[str, Any], seed: int,
            data: dict[str, Any], threads: int, save_dir: Path | None = None) -> dict[str, Any]:
    start = time.time()
    estimator = make_estimator(method, hp, seed=seed, threads=threads)
    estimator = fit_estimator(estimator, method, hp, seed, data)

    val_info = source_val_macro(estimator, data["val_per_domain"])
    val_scores = predict_scores(estimator, data["X_val"])
    in_scores = predict_scores(estimator, data["X_in"])
    cross_scores = predict_scores(estimator, data["X_cross"])
    in_metrics = metrics_from_scores(data["y_in"], in_scores)
    cross_metrics = metrics_from_scores(data["y_cross"], cross_scores)

    result = {
        "status": "success",
        "method": method,
        "target": target,
        "hp_id": hp_id,
        "seed": seed,
        "hparams": hp,
        "val_attack_f1_macro": val_info["val_attack_f1_macro"],
        "val_auroc_macro": val_info.get("val_auroc_macro"),
        "val_auprc_macro": val_info.get("val_auprc_macro"),
        "val_attack_f1_per_domain": val_info["val_attack_f1_per_domain"],
        "val_metrics_per_domain": val_info.get("val_metrics_per_domain"),
        "in_domain": in_metrics,
        "cross_domain": cross_metrics,
        "degradation": in_metrics["attack_f1"] - cross_metrics["attack_f1"],
        "fit_time_sec": time.time() - start,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "target_scores.npy", cross_scores.astype(np.float32))
        np.save(save_dir / "target_labels.npy", data["y_cross"].astype(np.int32))
        np.save(save_dir / "source_scores.npy", in_scores.astype(np.float32))
        np.save(save_dir / "source_labels.npy", data["y_in"].astype(np.int32))
        np.save(save_dir / "source_val_scores.npy", val_scores.astype(np.float32))
        np.save(save_dir / "source_val_labels.npy", data["y_val"].astype(np.int32))
        result["save_dir"] = str(save_dir)

    return result


def aggregate_search_records(method: str, target: str, hp_id: int, hp: dict[str, Any],
                             records: list[dict[str, Any]],
                             selection_metric: str) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "success"]
    if not ok:
        return {
            "status": "error",
            "method": method,
            "target": target,
            "hp_id": hp_id,
            "hparams": hp,
            "selection_metric": selection_metric,
            "error_message": "no successful search seeds",
        }

    def _mean(key: str) -> float | None:
        vals = [r.get(key) for r in ok]
        vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
        return float(np.mean(vals)) if vals else None

    def _mean_per_domain(key: str) -> dict[str, float]:
        domains = sorted({
            domain
            for record in ok
            for domain in (record.get(key) or {}).keys()
        })
        out = {}
        for domain in domains:
            vals = [
                (record.get(key) or {}).get(domain)
                for record in ok
            ]
            vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
            if vals:
                out[domain] = float(np.mean(vals))
        return out

    def _mean_metrics_per_domain() -> dict[str, dict[str, float]]:
        domains = sorted({
            domain
            for record in ok
            for domain in (record.get("val_metrics_per_domain") or {}).keys()
        })
        out = {}
        for domain in domains:
            metric_keys = sorted({
                metric
                for record in ok
                for metric in ((record.get("val_metrics_per_domain") or {}).get(domain) or {}).keys()
            })
            metrics = {}
            for metric in metric_keys:
                vals = [
                    ((record.get("val_metrics_per_domain") or {}).get(domain) or {}).get(metric)
                    for record in ok
                ]
                vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
                if vals:
                    metrics[metric] = float(np.mean(vals))
            if metrics:
                out[domain] = metrics
        return out

    out = {
        "status": "success",
        "method": method,
        "target": target,
        "hp_id": hp_id,
        "hparams": hp,
        "selection_metric": selection_metric,
        "search_seeds": [int(r["seed"]) for r in ok],
        "n_search_seeds_success": len(ok),
        "per_search_seed": [
            {
                "seed": r.get("seed"),
                "status": r.get("status"),
                selection_metric: r.get(selection_metric),
                "val_attack_f1_macro": r.get("val_attack_f1_macro"),
                "val_auroc_macro": r.get("val_auroc_macro"),
                "val_auprc_macro": r.get("val_auprc_macro"),
            }
            for r in records
        ],
    }
    for key in ["val_attack_f1_macro", "val_auroc_macro", "val_auprc_macro"]:
        out[key] = _mean(key)
    out["val_attack_f1_per_domain"] = _mean_per_domain("val_attack_f1_per_domain")
    out["val_metrics_per_domain"] = _mean_metrics_per_domain()
    return out


def summarize_final(final_records: list[dict[str, Any]], best: dict[str, Any]) -> dict[str, Any]:
    def mean_std(path: list[str]) -> tuple[float, float]:
        values = []
        for record in final_records:
            value = record
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    in_f1_m, in_f1_s = mean_std(["in_domain", "attack_f1"])
    cross_f1_m, cross_f1_s = mean_std(["cross_domain", "attack_f1"])
    in_auc_m, in_auc_s = mean_std(["in_domain", "auroc"])
    cross_auc_m, cross_auc_s = mean_std(["cross_domain", "auroc"])
    in_auprc_m, in_auprc_s = mean_std(["in_domain", "auprc"])
    cross_auprc_m, cross_auprc_s = mean_std(["cross_domain", "auprc"])
    deg_m, deg_s = mean_std(["degradation"])

    return {
        "status": "success",
        "method": final_records[0]["method"],
        "target": final_records[0]["target"],
        "hp_id": best["hp_id"],
        "hparams": best["hparams"],
        "selection_metric": best.get("selection_metric"),
        "search_seeds": best.get("search_seeds"),
        "n_search_seeds_success": best.get("n_search_seeds_success"),
        "search_selection_metric_value": best.get("selection_metric_value"),
        "seeds": [r["seed"] for r in final_records],
        "n_seeds": len(final_records),
        "in_attack_f1_mean": in_f1_m,
        "in_attack_f1_std": in_f1_s,
        "cross_attack_f1_mean": cross_f1_m,
        "cross_attack_f1_std": cross_f1_s,
        "in_auroc_mean": in_auc_m,
        "in_auroc_std": in_auc_s,
        "cross_auroc_mean": cross_auc_m,
        "cross_auroc_std": cross_auc_s,
        "in_auprc_mean": in_auprc_m,
        "in_auprc_std": in_auprc_s,
        "cross_auprc_mean": cross_auprc_m,
        "cross_auprc_std": cross_auprc_s,
        "degradation_mean": deg_m,
        "degradation_std": deg_s,
        "fit_time_sec_total": float(sum(r.get("fit_time_sec", 0.0) for r in final_records)),
    }


def run_method_target(method: str, target: str, data: dict[str, Any], args: argparse.Namespace) -> None:
    combo_dir = Path(args.out_dir) / f"{method}_{target}"
    search_dir = combo_dir / "search"
    final_dir = combo_dir / "final"
    search_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[tabular] method={method} target={target}")
    search_records = []
    selected_hp_items = select_hp_configs(method, args.n_search, target)
    if args.hp_ids is not None:
        selected_hp_items = [(hp_id, hp) for hp_id, hp in selected_hp_items if hp_id in args.hp_ids]

    for hp_id, hp in selected_hp_items:
        path = search_dir / f"hp{hp_id}.json"
        existing = load_json(path)
        if (
            existing is not None
            and existing.get("status") == "success"
            and existing.get("selection_metric") == args.selection_metric
            and existing.get("search_seeds") == args.search_seeds
            and existing.get(args.selection_metric) is not None
            and not args.force
        ):
            search_records.append(existing)
            print(f"  [skip] search hp{hp_id}")
            continue

        print(f"  [search] hp{hp_id} {hp}")
        seed_records = []
        for search_seed in args.search_seeds:
            seed_path = search_dir / f"hp{hp_id}_seed{search_seed}.json"
            existing_seed = load_json(seed_path)
            if existing_seed is not None and existing_seed.get("status") == "success" and not args.force:
                seed_records.append(existing_seed)
                print(f"    [skip] seed{search_seed}")
                continue
            try:
                record = run_one(method, target, hp_id, hp, seed=search_seed,
                                 data=data, threads=args.threads)
            except Exception as exc:
                record = {
                    "status": "error",
                    "method": method,
                    "target": target,
                    "hp_id": hp_id,
                    "seed": search_seed,
                    "hparams": hp,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            write_json(seed_path, record)
            seed_records.append(record)
        aggregate = aggregate_search_records(
            method, target, hp_id, hp, seed_records, args.selection_metric
        )
        write_json(path, aggregate)
        if aggregate.get("status") == "success":
            search_records.append(aggregate)

    if not search_records and args.final_seeds is None:
        raise RuntimeError(f"No successful tabular search runs for {method}_{target}")

    metric = args.selection_metric
    if args.final_seeds is not None:
        search_records = []
        for hp_id, _ in selected_hp_items:
            existing = load_json(search_dir / f"hp{hp_id}.json")
            if existing is not None and existing.get("status") == "success":
                search_records.append(existing)

    search_records = [r for r in search_records if r.get(metric) is not None]
    if not search_records:
        raise RuntimeError(f"No successful tabular search runs with selection metric {metric}")
    search_records.sort(key=lambda r: (-float(r[metric]), int(r["hp_id"])))
    best = {
        "method": method,
        "target": target,
        "hp_id": int(search_records[0]["hp_id"]),
        "hparams": search_records[0]["hparams"],
        "selection_metric": metric,
        "selection_metric_value": float(search_records[0][metric]),
        "search_seeds": search_records[0].get("search_seeds"),
        "n_search_seeds_success": search_records[0].get("n_search_seeds_success"),
        "val_attack_f1_macro": float(search_records[0].get("val_attack_f1_macro", float("nan"))),
        "val_auroc_macro": search_records[0].get("val_auroc_macro"),
        "val_auprc_macro": search_records[0].get("val_auprc_macro"),
        "val_attack_f1_per_domain": search_records[0].get("val_attack_f1_per_domain", {}),
        "val_metrics_per_domain": search_records[0].get("val_metrics_per_domain", {}),
    }
    write_json(final_dir / "best_hp.json", best)
    print(f"  [best] hp{best['hp_id']} {metric}={best['selection_metric_value']:.4f}")

    if args.search_only:
        return

    final_records = []
    final_seed_list = args.final_seeds if args.final_seeds is not None else args.seeds
    for seed in final_seed_list:
        path = final_dir / f"seed{seed}.json"
        artifact_dir = final_dir / f"seed{seed}_artifacts"
        existing = load_json(path)
        required_artifacts = [
            artifact_dir / "target_scores.npy",
            artifact_dir / "target_labels.npy",
            artifact_dir / "source_scores.npy",
            artifact_dir / "source_labels.npy",
            artifact_dir / "source_val_scores.npy",
            artifact_dir / "source_val_labels.npy",
        ]
        artifacts_complete = all(path.exists() and path.stat().st_size > 0 for path in required_artifacts)
        if existing is not None and existing.get("status") == "success" and artifacts_complete and not args.force:
            final_records.append(existing)
            print(f"  [skip] final seed{seed}")
            continue

        print(f"  [final] seed{seed}")
        try:
            record = run_one(
                method,
                target,
                best["hp_id"],
                best["hparams"],
                seed=seed,
                data=data,
                threads=args.threads,
                save_dir=artifact_dir,
            )
        except Exception as exc:
            record = {
                "status": "error",
                "method": method,
                "target": target,
                "hp_id": best["hp_id"],
                "seed": seed,
                "hparams": best["hparams"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json(path, record)
        if record.get("status") == "success":
            final_records.append(record)

    if args.final_seeds is None and len(final_records) != len(args.seeds):
        raise RuntimeError(f"Only {len(final_records)} final runs succeeded for {method}_{target}")

    if args.final_seeds is None:
        summary = summarize_final(final_records, best)
        write_json(final_dir / "summary.json", summary)
        print(
            f"  [summary] cross_f1={summary['cross_attack_f1_mean']:.4f} "
            f"cross_auroc={summary['cross_auroc_mean']:.4f}"
        )


def load_summaries(out_dir: str, methods: list[str], targets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        for target in targets:
            path = Path(out_dir) / f"{method}_{target}" / "final" / "summary.json"
            data = load_json(path)
            if data is not None and data.get("status") == "success":
                # Seed records may have been refreshed after the last summary.
                # Always aggregate the final records, never trust a stale mean.
                records = [load_json(path.parent / f"seed{s}.json") for s in data["seeds"]]
                best = load_json(path.parent / "best_hp.json")
                if best is None or any(r is None or r.get("status") != "success" for r in records):
                    raise ValueError(f"{method}/{target}: incomplete final seed records")
                data = summarize_final(records, best)
                write_json(path, data)
                rows.append(data)
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"[write] {path}")


def aggregate_tables(out_dir: str, aggregate_dir: str, methods: list[str], targets: list[str]) -> None:
    rows = load_summaries(out_dir, methods, targets)
    agg = Path(aggregate_dir)
    agg.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for d in rows:
        raw_rows.append({
            "method": d["method"],
            "target": d["target"],
            "hp_id": d["hp_id"],
            "cross_attack_f1_mean": d["cross_attack_f1_mean"],
            "cross_attack_f1_std": d["cross_attack_f1_std"],
            "cross_auroc_mean": d["cross_auroc_mean"],
            "cross_auroc_std": d["cross_auroc_std"],
            "cross_auprc_mean": d["cross_auprc_mean"],
            "cross_auprc_std": d["cross_auprc_std"],
            "in_attack_f1_mean": d["in_attack_f1_mean"],
            "in_auroc_mean": d["in_auroc_mean"],
            "in_auprc_mean": d["in_auprc_mean"],
            "degradation_mean": d["degradation_mean"],
        })

    keys = [
        "method", "target", "hp_id",
        "cross_attack_f1_mean", "cross_attack_f1_std",
        "cross_auroc_mean", "cross_auroc_std",
        "cross_auprc_mean", "cross_auprc_std",
        "in_attack_f1_mean", "in_auroc_mean", "in_auprc_mean",
        "degradation_mean",
    ]
    write_csv_rows(agg / "tabular_raw.csv", raw_rows, keys)

    # Compact, reproducible inputs for the public archive's aggregate checks.
    seed_rows = []
    for summary in rows:
        for seed in summary["seeds"]:
            path = Path(out_dir) / f"{summary['method']}_{summary['target']}" / "final" / f"seed{seed}.json"
            record = load_json(path)
            row = {k: record[k] for k in ("method", "target", "seed", "hp_id")}
            for prefix, key in (("cross", "cross_domain"), ("in", "in_domain")):
                for metric in ("attack_f1", "auroc", "auprc"):
                    row[f"{prefix}_{metric}"] = record[key][metric]
            row["degradation"] = record["degradation"]
            try:
                row["seed_record"] = path.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                row["seed_record"] = str(path)
            row["seed_record_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            seed_rows.append(row)
    if seed_rows:
        write_csv_rows(agg / "tabular_seed_metrics.csv", seed_rows, list(seed_rows[0]))

    by_combo = {(r["method"], r["target"]): r for r in raw_rows}
    lines = []
    lines.append("# Tabular Baseline Results")
    lines.append("")
    lines.append("Each cell shows mean +/- std over final seeds.")
    lines.append("")
    lines.append("## Cross-domain attack F1")
    lines.append("")
    lines.append("| Method | " + " | ".join(TARGET_DISPLAY[t] for t in targets) + " | Avg |")
    lines.append("|---|" + "|".join(["---"] * (len(targets) + 1)) + "|")
    for method in methods:
        values = []
        cells = []
        for target in targets:
            row = by_combo.get((method, target))
            if row is None:
                cells.append("--")
            else:
                values.append(float(row["cross_attack_f1_mean"]))
                cells.append(f"{row['cross_attack_f1_mean']:.4f} +/- {row['cross_attack_f1_std']:.3f}")
        avg = float(np.mean(values)) if values else float("nan")
        lines.append(f"| {METHOD_DISPLAY[method]} | " + " | ".join(cells) + f" | {avg:.4f} |")

    lines.append("")
    lines.append("## Cross-domain AUROC")
    lines.append("")
    lines.append("| Method | " + " | ".join(TARGET_DISPLAY[t] for t in targets) + " | Avg |")
    lines.append("|---|" + "|".join(["---"] * (len(targets) + 1)) + "|")
    for method in methods:
        values = []
        cells = []
        for target in targets:
            row = by_combo.get((method, target))
            if row is None:
                cells.append("--")
            else:
                values.append(float(row["cross_auroc_mean"]))
                cells.append(f"{row['cross_auroc_mean']:.4f} +/- {row['cross_auroc_std']:.3f}")
        avg = float(np.mean(values)) if values else float("nan")
        lines.append(f"| {METHOD_DISPLAY[method]} | " + " | ".join(cells) + f" | {avg:.4f} |")

    lines.append("")
    lines.append("## Cross-domain PR-AUC")
    lines.append("")
    lines.append("| Method | " + " | ".join(TARGET_DISPLAY[t] for t in targets) + " | Avg |")
    lines.append("|---|" + "|".join(["---"] * (len(targets) + 1)) + "|")
    for method in methods:
        values = []
        cells = []
        for target in targets:
            row = by_combo.get((method, target))
            if row is None:
                cells.append("--")
            else:
                values.append(float(row["cross_auprc_mean"]))
                cells.append(f"{row['cross_auprc_mean']:.4f} +/- {row['cross_auprc_std']:.3f}")
        avg = float(np.mean(values)) if values else float("nan")
        lines.append(f"| {METHOD_DISPLAY[method]} | " + " | ".join(cells) + f" | {avg:.4f} |")

    (agg / "Table_tabular_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {agg / 'Table_tabular_main.md'}")


def resample_by_prior(labels: np.ndarray, scores: np.ndarray, prior: float, seed: int,
                      max_n: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        return labels, scores

    n_total = int(min(len(pos) / max(prior, 1e-6), len(neg) / max(1.0 - prior, 1e-6), max_n))
    n_total = max(n_total, 2)
    n_pos = max(1, int(round(n_total * prior)))
    n_neg = max(1, n_total - n_pos)
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))

    rng = np.random.RandomState(seed)
    take = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ])
    rng.shuffle(take)
    return labels[take], scores[take]


def compute_tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_value: float) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_value = float(np.interp(fpr_value, fpr, tpr))
    threshold = float(np.interp(fpr_value, fpr, thresholds))
    return tpr_value, threshold


def write_posthoc_diagnostics(out_dir: str, aggregate_dir: str, methods: list[str], targets: list[str],
                              seeds: list[int]) -> None:
    agg = Path(aggregate_dir)
    prior_rows = []
    op_rows = []

    for method in methods:
        for target in targets:
            for seed in seeds:
                art = Path(out_dir) / f"{method}_{target}" / "final" / f"seed{seed}_artifacts"
                scores_path = art / "target_scores.npy"
                labels_path = art / "target_labels.npy"
                if not scores_path.exists() or not labels_path.exists():
                    continue
                scores = np.load(scores_path)
                labels = np.load(labels_path)

                for prior in PRIORS:
                    y_rs, s_rs = resample_by_prior(
                        labels, scores, prior, seed=stable_seed(method, target, seed, prior)
                    )
                    m = metrics_from_scores(y_rs, s_rs)
                    prior_rows.append({
                        "method": method,
                        "target": target,
                        "seed": seed,
                        "target_prior": prior,
                        "n_samples": len(y_rs),
                        "attack_ratio": m["attack_ratio"],
                        "f1": m["attack_f1"],
                        "auroc": m["auroc"],
                        "auprc": m["auprc"],
                    })

                base = metrics_from_scores(labels, scores)
                row = {
                    "method": method,
                    "target": target,
                    "seed": seed,
                    "n_samples": len(labels),
                    "attack_ratio": base["attack_ratio"],
                    "f1_at_0p5": base["attack_f1"],
                    "auroc": base["auroc"],
                    "auprc": base["auprc"],
                }
                for fpr_value in FPR_POINTS:
                    try:
                        tpr_value, threshold = compute_tpr_at_fpr(labels, scores, fpr_value)
                        alerts_per_million = float(np.mean(scores >= threshold) * 1_000_000.0)
                    except Exception:
                        tpr_value = float("nan")
                        threshold = float("nan")
                        alerts_per_million = float("nan")
                    tag = str(fpr_value).replace(".", "p")
                    row[f"tpr_at_fpr_{tag}"] = tpr_value
                    row[f"threshold_at_fpr_{tag}"] = threshold
                    row[f"alerts_per_million_at_fpr_{tag}"] = alerts_per_million
                op_rows.append(row)

    if prior_rows:
        write_csv_rows(
            agg / "tabular_prior_resample.csv",
            prior_rows,
            ["method", "target", "seed", "target_prior", "n_samples", "attack_ratio", "f1", "auroc", "auprc"],
        )
    if op_rows:
        keys = [
            "method", "target", "seed", "n_samples", "attack_ratio", "f1_at_0p5", "auroc", "auprc",
        ]
        for fpr_value in FPR_POINTS:
            tag = str(fpr_value).replace(".", "p")
            keys.extend([
                f"tpr_at_fpr_{tag}",
                f"threshold_at_fpr_{tag}",
                f"alerts_per_million_at_fpr_{tag}",
            ])
        write_csv_rows(agg / "tabular_operational.csv", op_rows, keys)

    summary = []
    summary.append("# Tabular Diagnostic Summary")
    summary.append("")
    summary.append("The tabular baselines are used to test whether the observed cross-domain failure is specific to the MLP backbone.")
    summary.append("Use the raw CSV files for exact values.")
    if prior_rows:
        summary.append("")
        summary.append(f"- Prior-resampling rows: {len(prior_rows)}")
    if op_rows:
        summary.append(f"- Operational metric rows: {len(op_rows)}")
    (agg / "tabular_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[write] {agg / 'tabular_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "DATA" / "nfv2_processed"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "RESULTS" / "model_family" / "tabular_baselines"))
    parser.add_argument("--aggregate_dir", default=str(PROJECT_ROOT / "RESULTS" / "model_family" / "tabular_aggregate"))
    parser.add_argument("--methods", default="all")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--seeds", default="default")
    parser.add_argument("--search_seeds", default="default")
    parser.add_argument("--n_search", type=int, default=4)
    parser.add_argument("--selection_metric", default="val_auroc_macro",
                        choices=["val_attack_f1_macro", "val_auroc_macro", "val_auprc_macro"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hp_ids", default=None)
    parser.add_argument("--final_seeds", default=None)
    parser.add_argument("--search_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    args.methods = parse_list(args.methods, METHODS)
    args.targets = parse_list(args.targets, TARGETS)
    args.seeds = parse_seeds(args.seeds, DEFAULT_SEEDS)
    args.search_seeds = parse_seeds(args.search_seeds, DEFAULT_SEARCH_SEEDS)
    args.hp_ids = None if args.hp_ids is None else sorted(set(parse_seeds(args.hp_ids, [])))
    args.final_seeds = None if args.final_seeds is None else parse_seeds(args.final_seeds, DEFAULT_SEEDS)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))

    if not args.aggregate_only:
        for target in args.targets:
            print(f"\n[load] LODO target={target}")
            data = load_lodo_arrays(args.data_dir, target)
            for method in args.methods:
                run_method_target(method, target, data, args)
                gc.collect()
            del data
            gc.collect()

    aggregate_tables(args.out_dir, args.aggregate_dir, args.methods, args.targets)
    write_posthoc_diagnostics(args.out_dir, args.aggregate_dir, args.methods, args.targets, args.seeds)


if __name__ == "__main__":
    main()
