"""Post-hoc NFv2 conditional-shift and exact label-conflict audit.

This script deliberately does not train, select, calibrate, or stop a model.  It
only reads the four persisted NFv2 domains, frozen group-safe splits, and frozen
raw39/relational_v1 prediction artifacts.
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from nidsbed.datasets import ALL_DOMAINS, load_domain_split, unique_row_weights
from nidsbed.model_selection import evaluate_score_arrays


PROTOCOL = "nfv2_conditional_label_shift_audit_v1"
DOMAINS = tuple(ALL_DOMAINS)
REPRESENTATIONS = ("raw39", "relational_v1")
SEEDS = (2026, 2027, 2028)
N_FEATURES = 39
MAX_SOURCE_CLASS_ROWS = 8000
MAX_TARGET_CLASS_ROWS = 16000
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              ensure_ascii=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def row_view(x: np.ndarray) -> np.ndarray:
    """Return collision-free fixed-width byte records for exact feature rows."""
    contiguous = np.ascontiguousarray(x)
    if contiguous.ndim != 2:
        raise ValueError("row_view expects a two-dimensional array")
    dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(dtype).reshape(-1)


def weighted_rate(mask: np.ndarray, weights: np.ndarray | None = None) -> float:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return float("nan")
    if weights is None:
        return float(mask.mean())
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(mask):
        raise ValueError("weight length mismatch")
    return float(np.sum(weights * mask) / np.sum(weights))


def _sample_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, int(limit), replace=False))


def source_class_sample(data_dir: Path, source_domains: tuple[str, ...],
                        cls: int, limit_per_domain: int, seed: int) -> np.ndarray:
    chunks = []
    for offset, domain in enumerate(source_domains):
        split = load_domain_split(data_dir, domain)
        y = np.load(data_dir / f"{domain}_y.npy", mmap_mode="r")
        X = np.load(data_dir / f"{domain}_X.npy", mmap_mode="r")
        candidates = np.asarray(split["train_idx"], dtype=np.int64)
        candidates = candidates[np.asarray(y[candidates], dtype=np.int64) == cls]
        chosen = _sample_indices(candidates, limit_per_domain, seed + 101 * offset)
        chunks.append(np.asarray(X[chosen], dtype=np.float32))
    if not chunks:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def build_source_index(data_dir: Path, target: str) -> dict:
    source_domains = tuple(domain for domain in DOMAINS if domain != target)
    key_parts, label_parts, sizes = [], [], []
    for domain in source_domains:
        split = load_domain_split(data_dir, domain)
        X = np.load(data_dir / f"{domain}_X.npy", mmap_mode="r")
        y = np.load(data_dir / f"{domain}_y.npy", mmap_mode="r")
        idx = np.asarray(split["train_idx"], dtype=np.int64)
        x_train = np.asarray(X[idx], dtype=np.float32)
        y_train = np.asarray(y[idx], dtype=np.int64)
        if x_train.ndim != 2 or x_train.shape[1] != N_FEATURES:
            raise ValueError(f"{domain}: expected source train shape N x 39")
        key_parts.append(row_view(x_train))
        label_parts.append(y_train)
        sizes.append(len(idx))
    all_keys = np.concatenate(key_parts)
    all_labels = np.concatenate(label_parts)
    unique_keys, inverse = np.unique(all_keys, return_inverse=True)
    label_bits = np.zeros(len(unique_keys), dtype=np.uint8)
    np.bitwise_or.at(label_bits, inverse, (1 << all_labels).astype(np.uint8))
    per_domain_bits = np.zeros((len(unique_keys), len(source_domains)), dtype=np.uint8)
    offset = 0
    for domain_i, size in enumerate(sizes):
        inv = inverse[offset:offset + size]
        labels = all_labels[offset:offset + size]
        np.bitwise_or.at(per_domain_bits[:, domain_i], inv,
                         (1 << labels).astype(np.uint8))
        offset += size
    return {
        "source_domains": list(source_domains),
        "unique_keys": unique_keys,
        "label_bits": label_bits,
        "per_domain_bits": per_domain_bits,
        "n_source_train_rows": int(len(all_keys)),
        "n_unique_source_feature_rows": int(len(unique_keys)),
    }


def lookup_source_bits(target_x: np.ndarray, source_index: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = row_view(np.asarray(target_x, dtype=np.float32))
    unique_keys = source_index["unique_keys"]
    positions = np.searchsorted(unique_keys, keys)
    matched = positions < len(unique_keys)
    safe = np.minimum(positions, max(0, len(unique_keys) - 1))
    if len(unique_keys):
        matched &= unique_keys[safe] == keys
    bits = np.zeros(len(keys), dtype=np.uint8)
    per_domain = np.zeros((len(keys), len(source_index["source_domains"])), dtype=np.uint8)
    bits[matched] = source_index["label_bits"][positions[matched]]
    per_domain[matched] = source_index["per_domain_bits"][positions[matched]]
    return bits, per_domain


def exact_conflict_summary(target_x: np.ndarray, target_y: np.ndarray,
                           target_groups: np.ndarray, source_index: dict) -> dict:
    target_y = np.asarray(target_y, dtype=np.int64)
    bits, per_domain = lookup_source_bits(target_x, source_index)
    matched = bits != 0
    same = (bits & (1 << target_y).astype(np.uint8)) != 0
    opposite = (bits & (1 << (1 - target_y)).astype(np.uint8)) != 0
    only_opposite = matched & opposite & ~same
    mixed = bits == 3
    weights = unique_row_weights(np.asarray(target_x, dtype=np.float32))
    out = {
        "n_rows": int(len(target_y)),
        "n_unique_target_groups": int(np.unique(target_groups).size),
        "raw_row": {
            "match_rate": weighted_rate(matched),
            "any_opposite_label_rate": weighted_rate(opposite),
            "only_opposite_label_rate": weighted_rate(only_opposite),
            "mixed_source_labels_rate": weighted_rate(mixed),
        },
        "duplicate_group_equal": {
            "match_rate": weighted_rate(matched, weights),
            "any_opposite_label_rate": weighted_rate(opposite, weights),
            "only_opposite_label_rate": weighted_rate(only_opposite, weights),
            "mixed_source_labels_rate": weighted_rate(mixed, weights),
        },
        "per_source_raw_row": {},
    }
    for i, domain in enumerate(source_index["source_domains"]):
        domain_bits = per_domain[:, i]
        dm = domain_bits != 0
        do = (domain_bits & (1 << (1 - target_y)).astype(np.uint8)) != 0
        ds = (domain_bits & (1 << target_y).astype(np.uint8)) != 0
        out["per_source_raw_row"][domain] = {
            "match_rate": weighted_rate(dm),
            "any_opposite_label_rate": weighted_rate(do),
            "only_opposite_label_rate": weighted_rate(dm & do & ~ds),
        }
    return out


def conditional_feature_shift(source_x: np.ndarray, target_x: np.ndarray,
                              feature_names: list[str]) -> dict:
    if source_x.ndim != 2 or target_x.ndim != 2 or source_x.shape[1] != len(feature_names):
        raise ValueError("feature shift input shape mismatch")
    per_feature = []
    for j, name in enumerate(feature_names):
        s = np.asarray(source_x[:, j], dtype=np.float64)
        t = np.asarray(target_x[:, j], dtype=np.float64)
        sf = np.isfinite(s)
        tf = np.isfinite(t)
        if sf.any() and tf.any():
            sm = float(np.mean(s[sf])); tm = float(np.mean(t[tf]))
            sv = float(np.var(s[sf])); tv = float(np.var(t[tf]))
            smd = abs(tm - sm) / np.sqrt(0.5 * (sv + tv) + 1e-12)
        else:
            smd = None
        per_feature.append({
            "feature": name,
            "abs_smd": finite_or_none(smd),
            "missingness_delta": float(np.mean(~tf) - np.mean(~sf)),
            "source_finite_fraction": float(np.mean(sf)),
            "target_finite_fraction": float(np.mean(tf)),
        })
    valid_smd = [x["abs_smd"] for x in per_feature if x["abs_smd"] is not None]
    ranked = sorted((x for x in per_feature if x["abs_smd"] is not None),
                    key=lambda x: (-x["abs_smd"], x["feature"]))
    return {
        "n_source_rows": int(len(source_x)),
        "n_target_rows": int(len(target_x)),
        "mean_abs_smd": finite_or_none(np.mean(valid_smd)) if valid_smd else None,
        "median_abs_smd": finite_or_none(np.median(valid_smd)) if valid_smd else None,
        "max_abs_smd": finite_or_none(np.max(valid_smd)) if valid_smd else None,
        "mean_abs_missingness_delta": float(np.mean([abs(x["missingness_delta"]) for x in per_feature])),
        "top_features": ranked[:10],
        "per_feature": per_feature,
    }


def score_distribution(scores: np.ndarray, labels: np.ndarray, cls: int) -> dict:
    values = np.asarray(scores, dtype=np.float64)[np.asarray(labels) == cls]
    if len(values) == 0:
        return {"n": 0, "mean": None, "std": None, "quantiles": {str(q): None for q in QUANTILES}}
    qs = np.quantile(values, QUANTILES)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, qs)},
    }


def _metric_value(metrics: dict, key: str) -> float | None:
    return finite_or_none(metrics.get(key))


def _feature_names(data_dir: Path) -> list[str]:
    names = [x.strip() for x in (data_dir / "feature_names.txt").read_text().splitlines() if x.strip()]
    if len(names) != N_FEATURES:
        raise ValueError(f"expected 39 feature names, found {len(names)}")
    return names


def _load_coordinate(model_root: Path, target: str, representation: str, seed: int) -> tuple[dict, dict]:
    base = model_root / "model" / f"target_{target}" / representation / f"seed_{seed}"
    result_path = base / "result.json"
    pred_path = base / "predictions.npz"
    if not result_path.is_file() or not pred_path.is_file():
        raise ValueError(f"missing frozen model coordinate: {base}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    pred = dict(np.load(pred_path, allow_pickle=False))
    required = {"target_scores", "target_labels", "target_group_ids",
                "source_val_scores", "source_val_labels", "source_val_group_ids"}
    if set(pred) != required:
        raise ValueError(f"{base}: prediction keys mismatch: {set(pred)}")
    return result, pred


def _source_val_prior(data_dir: Path, source_domains: tuple[str, ...]) -> dict:
    rates = {}
    rows = 0
    attacks = 0
    for domain in source_domains:
        split = load_domain_split(data_dir, domain)
        y = np.asarray(np.load(data_dir / f"{domain}_y.npy", mmap_mode="r")[split["train_idx"]], dtype=np.int64)
        rates[domain] = float(y.mean())
        rows += len(y); attacks += int(y.sum())
    return {"per_domain": rates, "pooled": float(attacks / rows), "n_rows": int(rows), "n_attack": attacks}


def analyze_target(data_dir: Path, model_root: Path, target: str,
                   output_dir: Path, feature_names: list[str]) -> dict:
    source_domains = tuple(domain for domain in DOMAINS if domain != target)
    target_X = np.load(data_dir / f"{target}_X.npy", mmap_mode="r")
    target_y_full = np.load(data_dir / f"{target}_y.npy", mmap_mode="r")
    split = load_domain_split(data_dir, target)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    target_x = np.asarray(target_X[test_idx], dtype=np.float32)
    target_y = np.asarray(target_y_full[test_idx], dtype=np.int64)
    target_groups = np.asarray(split["row_group_id"][test_idx], dtype=np.int64)
    if len(target_x) == 0 or not np.isin(target_y, [0, 1]).all():
        raise ValueError(f"{target}: invalid fixed target test")

    source_prior = _source_val_prior(data_dir, source_domains)
    source_index = build_source_index(data_dir, target)
    conflict = exact_conflict_summary(target_x, target_y, target_groups, source_index)
    feature_shift = {}
    for cls, name in ((0, "benign"), (1, "attack")):
        sx = source_class_sample(data_dir, source_domains, cls, MAX_SOURCE_CLASS_ROWS, 7000 + 100 * DOMAINS.index(target) + cls)
        tx_idx = _sample_indices(np.flatnonzero(target_y == cls), MAX_TARGET_CLASS_ROWS,
                                 8000 + 100 * DOMAINS.index(target) + cls)
        feature_shift[name] = conditional_feature_shift(sx, target_x[tx_idx], feature_names)

    rows = []
    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            result, pred = _load_coordinate(model_root, target, representation, seed)
            p_y = np.asarray(pred["target_labels"], dtype=np.int64)
            p_groups = np.asarray(pred["target_group_ids"], dtype=np.int64)
            p_scores = np.asarray(pred["target_scores"], dtype=np.float64)
            if not np.array_equal(p_y, target_y) or not np.array_equal(p_groups, target_groups):
                raise ValueError(f"{target}/{representation}/seed{seed}: prediction labels/groups do not match frozen split")
            source_scores = np.asarray(pred["source_val_scores"], dtype=np.float64)
            source_labels = np.asarray(pred["source_val_labels"], dtype=np.int64)
            metrics = result["metrics"]
            target_benign = score_distribution(p_scores, p_y, 0)
            target_attack = score_distribution(p_scores, p_y, 1)
            source_benign = score_distribution(source_scores, source_labels, 0)
            source_attack = score_distribution(source_scores, source_labels, 1)
            row = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "outer_target": target,
                "representation": representation,
                "seed": int(seed),
                "source_domains": list(source_domains),
                "target_labels_used_for_training_or_selection": False,
                "target_labels_read_posthoc": True,
                "target_test": {
                    "n_rows": int(len(target_y)),
                    "attack_rate": float(target_y.mean()),
                    "n_unique_groups": int(np.unique(target_groups).size),
                },
                "source_train": source_prior,
                "prior_shift": float(target_y.mean() - source_prior["pooled"]),
                "score_distributions": {
                    "source_validation": {"benign": source_benign, "attack": source_attack},
                    "target_test": {"benign": target_benign, "attack": target_attack},
                },
                "score_orientation": {
                    "source_attack_minus_benign_mean": finite_or_none((source_attack["mean"] or 0.0) - (source_benign["mean"] or 0.0)),
                    "target_attack_minus_benign_mean": finite_or_none((target_attack["mean"] or 0.0) - (target_benign["mean"] or 0.0)),
                    "target_attack_below_benign_mean": bool((target_attack["mean"] or 0.0) < (target_benign["mean"] or 0.0)),
                },
                "metrics": {
                    "row_auroc": _metric_value(metrics, "auroc"),
                    "group_auroc": _metric_value(metrics, "unique_auroc"),
                    "row_auprc": _metric_value(metrics, "auprc"),
                    "group_auprc": _metric_value(metrics, "unique_auprc"),
                    "row_partial_auc_1pct": _metric_value(metrics, "partial_auc_1pct"),
                    "group_partial_auc_1pct": _metric_value(metrics, "unique_partial_auc_1pct"),
                    "row_recall_at_source_threshold": _metric_value(metrics, "source_threshold_recall_row"),
                    "group_recall_at_source_threshold": _metric_value(metrics, "source_threshold_recall_group"),
                    "row_false_alerts_per_million_benign": _metric_value(metrics, "source_threshold_false_alerts_m_row"),
                    "group_false_alerts_per_million_benign": _metric_value(metrics, "source_threshold_false_alerts_m_group"),
                    "row_brier": _metric_value(metrics, "brier_row"),
                    "group_brier": _metric_value(metrics, "brier_group"),
                    "row_ece": _metric_value(metrics, "ece_row"),
                },
                "mechanism_summary": {
                    "exact_feature_label_conflict": conflict,
                    "conditional_feature_shift": feature_shift,
                },
                "frozen_model_artifact": {
                    "result_sha256": sha256_file(model_root / "model" / f"target_{target}" / representation / f"seed_{seed}" / "result.json"),
                    "predictions_sha256": sha256_file(model_root / "model" / f"target_{target}" / representation / f"seed_{seed}" / "predictions.npz"),
                    "source_model_code_sha256": result.get("code_sha256"),
                    "source_model_config_sha256": result.get("config_sha256"),
                },
            }
            rows.append(row)
            out_path = output_dir / "coordinates" / f"target_{target}" / representation / f"seed_{seed}.json"
            atomic_write_json(out_path, row)
    target_audit = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "target": target,
        "source_domains": list(source_domains),
        "target_labels_used_for_training_or_selection": False,
        "target_labels_read_posthoc": True,
        "fixed_test": {
            "n_rows": int(len(target_y)),
            "attack_rate": float(target_y.mean()),
            "n_unique_groups": int(np.unique(target_groups).size),
        },
        "source_train": source_prior,
        "exact_feature_label_conflict": conflict,
        "conditional_feature_shift": feature_shift,
        "sampling": {
            "max_source_rows_per_class_per_domain": MAX_SOURCE_CLASS_ROWS,
            "max_target_rows_per_class": MAX_TARGET_CLASS_ROWS,
            "deterministic": True,
        },
        "n_coordinates": len(rows),
    }
    atomic_write_json(output_dir / "targets" / f"{target}.json", target_audit)
    del source_index, target_X, target_y_full
    gc.collect()
    return {"target": target, "rows": rows, "target_audit": target_audit}


def _manifest(data_dir: Path, model_root: Path, output_dir: Path,
              feature_names: list[str]) -> dict:
    inputs = []
    for domain in DOMAINS:
        for suffix in ("X", "y", "attack"):
            p = data_dir / f"{domain}_{suffix}.npy"
            inputs.append({"path": str(p.resolve()), "sha256": sha256_file(p), "bytes": p.stat().st_size})
        p = data_dir / "splits" / f"{domain}_split.npz"
        inputs.append({"path": str(p.resolve()), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    predictions = []
    for target in DOMAINS:
        for rep in REPRESENTATIONS:
            for seed in SEEDS:
                base = model_root / "model" / f"target_{target}" / rep / f"seed_{seed}"
                for name in ("result.json", "predictions.npz"):
                    p = base / name
                    predictions.append({"path": str(p.resolve()), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    code_path = Path(__file__).resolve()
    fingerprint_path = model_root / "aggregate" / "domain_summary.csv"
    if not fingerprint_path.is_file():
        raise ValueError(f"required fixed domain-fingerprint summary is missing: {fingerprint_path}")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "domains": list(DOMAINS),
        "representations": list(REPRESENTATIONS),
        "seeds": list(SEEDS),
        "n_features": N_FEATURES,
        "feature_names": feature_names,
        "external_data_allowed": False,
        "target_labels_used_for_training_or_selection": False,
        "target_labels_read_posthoc": True,
        "source_prediction_root": str(model_root.resolve()),
        "input_files": inputs,
        "frozen_prediction_files": predictions,
        "fixed_domain_fingerprint_summary": {
            "path": str(fingerprint_path.resolve()),
            "sha256": sha256_file(fingerprint_path),
            "role": "descriptive relation input only; no target labels or selection",
        },
        "audit_code": {"path": str(code_path), "sha256": sha256_file(code_path)},
        "sampling": {"max_source_rows_per_class_per_domain": MAX_SOURCE_CLASS_ROWS,
                      "max_target_rows_per_class": MAX_TARGET_CLASS_ROWS,
                      "quantiles": list(QUANTILES)},
        "manifest_content_sha256": canonical_sha256({"inputs": inputs, "predictions": predictions, "feature_names": feature_names}),
    }


def run(data_dir: Path, model_root: Path, output_dir: Path) -> dict:
    data_dir = data_dir.resolve(); model_root = model_root.resolve(); output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = _feature_names(data_dir)
    manifest = _manifest(data_dir, model_root, output_dir, feature_names)
    atomic_write_json(output_dir / "protocol_manifest.json", manifest)
    results = []
    for target in DOMAINS:
        print(f"[{target}] source index, exact conflicts, and conditional shifts", flush=True)
        results.append(analyze_target(data_dir, model_root, target, output_dir, feature_names))
    status = {"schema_version": 1, "protocol": PROTOCOL, "status": "success",
              "n_targets": len(results), "n_coordinates": sum(len(x["rows"]) for x in results),
              "external_data_used": False, "target_labels_used_for_training_or_selection": False,
              "target_labels_read_posthoc": True,
              "manifest_content_sha256": manifest["manifest_content_sha256"]}
    atomic_write_json(output_dir / "run_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "DATA" / "nfv2_processed"))
    parser.add_argument("--model_root", default=str(PROJECT_ROOT / "RESULTS" / "diagnostics" / "nfv2_domain_fingerprint_relational_v1"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "RESULTS" / "diagnostics" / "nfv2_conditional_label_shift_audit_v1"))
    args = parser.parse_args()
    status = run(Path(args.data_dir), Path(args.model_root), Path(args.output_dir))
    print(json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
