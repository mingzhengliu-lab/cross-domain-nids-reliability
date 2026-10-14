"""NFv2 source-only domain-fingerprint and relational-v1 audit.

This runner deliberately keeps target labels out of preprocessing, training,
checkpoint selection, and domain-classifier selection.  Target labels are
opened only by the frozen ERM evaluator after source thresholds are fixed.
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from nidsbed.calibration import binary_ece
from nidsbed.datasets import (
    ALL_DOMAINS,
    GPUResidentBatchIterator,
    apply_source_train_preprocessor,
    fit_source_train_preprocessor,
    load_domain_split,
)
from nidsbed.model_selection import (
    evaluate_score_arrays,
    metrics_at_fixed_threshold,
    predict_logits_batched,
    select_threshold_at_max_fpr,
)
from nidsbed.lib.misc import seed_everything
from nidsbed.diagnostic_helpers import (
    combine_domain_group_ids,
    evaluate_model,
    group_weights,
    train_source_erm,
)
from nidsbed.diagnostic_helpers import (
    select_weighted_threshold_at_max_fpr,
    weighted_fixed_threshold,
)


DATA_DIR = PROJECT_ROOT / "DATA" / "nfv2_processed"
OUTPUT_ROOT = PROJECT_ROOT / "RESULTS" / "diagnostics" / "nfv2_domain_fingerprint_relational_v1"
PROTOCOL_VERSION = "nfv2_domain_fingerprint_relational_v1"
DOMAINS = tuple(ALL_DOMAINS)
SEEDS = (2026, 2027, 2028)
REPRESENTATIONS = ("raw39", "relational_v1")
DOMAIN_TASKS = ("all", "benign", "attack")
DOMAIN_MODELS = ("logreg", "mlp")
RAW_FEATURE_COUNT = 39
MAX_DOMAIN_GROUPS = 30_000
PERMUTATION_MAX_ROWS = 6_000
DOMAIN_MLP_HIDDEN = (128, 64)
DOMAIN_MLP_DROPOUT = 0.2
DOMAIN_MLP_MAX_EPOCHS = 30
DOMAIN_MLP_PATIENCE = 5
DOMAIN_MLP_BATCH = 2048
ERM_MAX_EPOCHS = 30
ERM_PATIENCE = 6


RELATIONAL_SPECS = (
    ("log1p_IN_BYTES", "log1p", ("IN_BYTES",)),
    ("log1p_OUT_BYTES", "log1p", ("OUT_BYTES",)),
    ("log1p_IN_PKTS", "log1p", ("IN_PKTS",)),
    ("log1p_OUT_PKTS", "log1p", ("OUT_PKTS",)),
    ("log1p_FLOW_DURATION_MILLISECONDS", "log1p", ("FLOW_DURATION_MILLISECONDS",)),
    ("log1p_DURATION_IN", "log1p", ("DURATION_IN",)),
    ("log1p_DURATION_OUT", "log1p", ("DURATION_OUT",)),
    ("log1p_SRC_TO_DST_AVG_THROUGHPUT", "log1p", ("SRC_TO_DST_AVG_THROUGHPUT",)),
    ("log1p_DST_TO_SRC_AVG_THROUGHPUT", "log1p", ("DST_TO_SRC_AVG_THROUGHPUT",)),
    ("in_byte_share", "share", ("IN_BYTES", "OUT_BYTES")),
    ("in_pkt_share", "share", ("IN_PKTS", "OUT_PKTS")),
    ("duration_balance", "balance", ("DURATION_IN", "DURATION_OUT")),
    ("throughput_balance", "balance", ("SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT")),
    ("retransmit_in_rate", "rate", ("RETRANSMITTED_IN_PKTS", "IN_PKTS")),
    ("retransmit_out_rate", "rate", ("RETRANSMITTED_OUT_PKTS", "OUT_PKTS")),
    ("in_bytes_per_pkt_log", "logdiff", ("IN_BYTES", "IN_PKTS")),
    ("out_bytes_per_pkt_log", "logdiff", ("OUT_BYTES", "OUT_PKTS")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return sha256_bytes(arr.view(np.uint8).tobytes())


def protocol_config() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "domains": list(DOMAINS),
        "representations": list(REPRESENTATIONS),
        "domain_tasks": list(DOMAIN_TASKS),
        "domain_models": list(DOMAIN_MODELS),
        "seeds": list(SEEDS),
        "max_domain_groups": MAX_DOMAIN_GROUPS,
        "permutation_max_rows": PERMUTATION_MAX_ROWS,
        "domain_mlp_hidden": list(DOMAIN_MLP_HIDDEN),
        "domain_mlp_dropout": DOMAIN_MLP_DROPOUT,
        "domain_mlp_max_epochs": DOMAIN_MLP_MAX_EPOCHS,
        "domain_mlp_patience": DOMAIN_MLP_PATIENCE,
        "domain_mlp_batch": DOMAIN_MLP_BATCH,
        "erm_max_epochs": ERM_MAX_EPOCHS,
        "erm_patience": ERM_PATIENCE,
        "relational_specs": [list(item) for item in RELATIONAL_SPECS],
    }


def code_fingerprint() -> str:
    """Hash this implementation, its shared helpers and the embedded protocol."""
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), PROJECT_ROOT / "CODE/common/nidsbed/diagnostic_helpers.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps(protocol_config(), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def config_fingerprint() -> str:
    return sha256_bytes(json.dumps(protocol_config(), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def feature_names(data_dir: Path = DATA_DIR) -> list[str]:
    names = [x.strip() for x in (data_dir / "feature_names.txt").read_text().splitlines() if x.strip()]
    if len(names) != RAW_FEATURE_COUNT:
        raise ValueError(f"expected {RAW_FEATURE_COUNT} raw features, found {len(names)}")
    return names


def relational_feature_names(names: list[str]) -> list[str]:
    required = {source for _, _, sources in RELATIONAL_SPECS for source in sources}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"relational-v1 fields missing from raw39: {missing}")
    return list(names) + [name for name, _, _ in RELATIONAL_SPECS]


def _safe_nonnegative(X: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X[:, index], dtype=np.float64)
    valid = np.isfinite(values) & (np.abs(values) < 1e9) & (values >= 0.0)
    safe = np.where(valid, values, np.nan)
    return safe, valid


def relational_transform_raw(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Append the pre-registered relational-v1 features to raw39.

    Invalid or semantically negative source values propagate as NaN so the
    existing source-fitted invalid-indicator policy remains active.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X must be a two-dimensional raw feature matrix")
    lookup = {name: idx for idx, name in enumerate(names)}
    derived = []
    for _, kind, sources in RELATIONAL_SPECS:
        values = [_safe_nonnegative(X, lookup[source]) for source in sources]
        a, va = values[0]
        if kind == "log1p":
            value = np.log1p(a)
            valid = va
        elif kind == "share":
            b, vb = values[1]
            value = a / (a + b + 1.0)
            valid = va & vb
        elif kind == "balance":
            b, vb = values[1]
            value = (a - b) / (a + b + 1.0)
            valid = va & vb
        elif kind == "rate":
            b, vb = values[1]
            value = a / (b + 1.0)
            valid = va & vb
        elif kind == "logdiff":
            b, vb = values[1]
            value = np.log1p(a) - np.log1p(b)
            valid = va & vb
        else:  # pragma: no cover
            raise AssertionError(kind)
        derived.append(np.where(valid, value, np.nan).astype(np.float32))
    return np.column_stack([np.asarray(X, dtype=np.float32), *derived]).astype(np.float32), relational_feature_names(names)


def transform_raw(X: np.ndarray, names: list[str], representation: str) -> tuple[np.ndarray, list[str]]:
    if representation == "raw39":
        return np.asarray(X, dtype=np.float32), list(names)
    if representation == "relational_v1":
        return relational_transform_raw(X, names)
    raise ValueError(f"unknown representation {representation!r}")


def _representative_indices(group_ids: np.ndarray, max_groups: int, seed: int) -> np.ndarray:
    groups = np.asarray(group_ids)
    _, first = np.unique(groups, return_index=True)
    first = np.sort(first)
    if len(first) <= max_groups:
        return first.astype(np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(first, max_groups, replace=False)).astype(np.int64)


def _task_representative_indices(group_ids: np.ndarray, y: np.ndarray, task: str | None,
                                 max_groups: int, seed: int) -> np.ndarray:
    """Select group representatives after applying a domain-task filter.

    Filtering before grouping is important for benign/attack domain tasks:
    the first row of an exact feature group may carry the opposite label.
    """
    groups = np.asarray(group_ids)
    labels = np.asarray(y, dtype=np.int64)
    if task is None or task == "all":
        mask = np.ones(len(labels), dtype=bool)
    elif task == "benign":
        mask = labels == 0
    elif task == "attack":
        mask = labels == 1
    else:
        raise ValueError(f"unknown task {task!r}")
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    candidate_groups = groups[candidates]
    _, first = np.unique(candidate_groups, return_index=True)
    representatives = np.sort(candidates[first]).astype(np.int64)
    if len(representatives) <= max_groups:
        return representatives
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(representatives, max_groups, replace=False)).astype(np.int64)


def _binary_group_representative_indices(group_ids: np.ndarray, y: np.ndarray,
                                         max_groups: int, seed: int) -> np.ndarray:
    """Select one row per exact group while preserving both labels when possible.

    This helper is used only by the small model smoke view. The full protocol keeps
    the existing representative rule and therefore does not change its estimand.
    """
    groups = np.asarray(group_ids)
    labels = np.asarray(y, dtype=np.int64)
    _, inverse = np.unique(groups, return_inverse=True)
    candidates: list[list[int]] = [[] for _ in range(int(inverse.max()) + 1)] if len(inverse) else []
    for idx, group_index in enumerate(inverse):
        candidates[int(group_index)].append(idx)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    chosen: list[int] = []
    counts = {0: 0, 1: 0}
    for group_index in order:
        rows = candidates[int(group_index)]
        preferred_label = 0 if counts[0] <= counts[1] else 1
        preferred = [idx for idx in rows if labels[idx] == preferred_label]
        idx = preferred[0] if preferred else rows[0]
        chosen.append(idx)
        counts[int(labels[idx])] = counts.get(int(labels[idx]), 0) + 1
    chosen_array = np.asarray(chosen, dtype=np.int64)
    if len(chosen_array) <= max_groups:
        return np.sort(chosen_array)
    selected: list[int] = []
    per_class = max_groups // 2
    for label in (0, 1):
        pool = chosen_array[labels[chosen_array] == label]
        take = min(len(pool), per_class)
        if take:
            selected.extend(rng.choice(pool, take, replace=False).tolist())
    remaining = max_groups - len(selected)
    if remaining:
        pool = np.setdiff1d(chosen_array, np.asarray(selected, dtype=np.int64), assume_unique=False)
        selected.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def _task_row_indices(y: np.ndarray, task: str | None, max_rows: int | None, seed: int) -> np.ndarray:
    """Filter validation rows by task without collapsing repeated validation rows."""
    labels = np.asarray(y, dtype=np.int64)
    if task is None or task == "all":
        candidates = np.arange(len(labels), dtype=np.int64)
    elif task == "benign":
        candidates = np.flatnonzero(labels == 0).astype(np.int64)
    elif task == "attack":
        candidates = np.flatnonzero(labels == 1).astype(np.int64)
    else:
        raise ValueError(f"unknown task {task!r}")
    if max_rows is None or len(candidates) <= max_rows:
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, int(max_rows), replace=False)).astype(np.int64)


def _limit_group_rows(X: np.ndarray, y: np.ndarray, groups: np.ndarray, limit: int | None, seed: int):
    if limit is None or len(X) <= limit:
        return X, y, groups
    idx = _representative_indices(groups, int(limit), seed)
    return X[idx], y[idx], groups[idx]


def _preprocess_view(
    data_dir: Path,
    train_domains: tuple[str, ...],
    eval_domain: str,
    representation: str,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_target_rows: int | None = None,
    domain_task: str | None = None,
    preserve_binary_classes: bool = False,
) -> dict:
    names = feature_names(data_dir)
    raw_train: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    raw_val: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    source_pool = []
    split_hash_parts = []
    overlap_audit = {}
    for offset, domain in enumerate(train_domains):
        split = load_domain_split(data_dir, domain)
        split_path = data_dir / "splits" / f"{domain}_split.npz"
        split_hash_parts.append(sha256_file(split_path))
        X = np.load(data_dir / f"{domain}_X.npy", mmap_mode="r")
        y = np.load(data_dir / f"{domain}_y.npy", mmap_mode="r").astype(np.int64)
        tr = np.asarray(split["train_idx"], dtype=np.int64)
        va = np.asarray(split["val_idx"], dtype=np.int64)
        te = np.asarray(split["test_idx"], dtype=np.int64)
        train_groups = np.asarray(split["row_group_id"][tr])
        val_groups = np.asarray(split["row_group_id"][va])
        test_groups = np.asarray(split["row_group_id"][te])
        overlap_audit[domain] = {
            "train_val": int(len(np.intersect1d(np.unique(train_groups), np.unique(val_groups)))),
            "train_test": int(len(np.intersect1d(np.unique(train_groups), np.unique(test_groups)))),
            "val_test": int(len(np.intersect1d(np.unique(val_groups), np.unique(test_groups)))),
        }
        if domain_task is None and not preserve_binary_classes:
            all_tr_rep = np.arange(len(tr), dtype=np.int64)
            all_va_rep = np.arange(len(va), dtype=np.int64)
        elif preserve_binary_classes:
            all_tr_rep = _binary_group_representative_indices(
                split["row_group_id"][tr], y[tr], max_train_rows or len(tr), 9100 + offset
            )
            all_va_rep = _binary_group_representative_indices(
                split["row_group_id"][va], y[va], max_val_rows or len(va), 9200 + offset
            )
        else:
            all_tr_rep = _representative_indices(split["row_group_id"][tr], max_train_rows or len(tr), 9100 + offset)
            all_va_rep = _representative_indices(split["row_group_id"][va], max_val_rows or len(va), 9200 + offset)
        task_tr_rep = _task_representative_indices(
            split["row_group_id"][tr], y[tr], domain_task, max_train_rows or len(tr), 9150 + offset
        ) if domain_task is not None else all_tr_rep
        if domain_task is not None and not preserve_binary_classes:
            # Validation is already group-disjoint from training. Retaining all
            # task-filtered rows avoids a one-group class collapsing a domain's
            # validation metric (notably ToN benign validation).
            task_va_rep = _task_row_indices(y[va], domain_task, max_val_rows, 9250 + offset)
        else:
            task_va_rep = _task_representative_indices(
                split["row_group_id"][va], y[va], domain_task, max_val_rows or len(va), 9250 + offset
            ) if domain_task is not None else all_va_rep
        all_tr_idx = tr[all_tr_rep]
        tr_idx = tr[task_tr_rep]
        va_idx = va[task_va_rep]
        pool_raw, pool_names = transform_raw(np.asarray(X[all_tr_idx]), names, representation)
        tr_raw, tr_names = transform_raw(np.asarray(X[tr_idx]), names, representation)
        va_raw, va_names = transform_raw(np.asarray(X[va_idx]), names, representation)
        if tr_names != va_names:
            raise AssertionError("representation names differ across splits")
        tr_tuple = (tr_raw, y[tr_idx], np.asarray(split["row_group_id"][tr_idx]))
        va_tuple = (va_raw, y[va_idx], np.asarray(split["row_group_id"][va_idx]))
        raw_train[domain] = tr_tuple
        raw_val[domain] = va_tuple
        source_pool.append(pool_raw)
    selected_names = tr_names
    prep = fit_source_train_preprocessor(np.concatenate(source_pool), feature_names=selected_names)
    capped_pool = np.concatenate([apply_source_train_preprocessor(x, prep)[0] for x in source_pool])
    scaler = StandardScaler().fit(capped_pool)

    def apply(X: np.ndarray) -> np.ndarray:
        capped, invalid = apply_source_train_preprocessor(X, prep)
        transformed = np.clip(scaler.transform(capped), -10.0, 10.0).astype(np.float32)
        return np.concatenate([transformed, invalid], axis=1).astype(np.float32, copy=False)

    train = {domain: (apply(x), y, g) for domain, (x, y, g) in raw_train.items()}
    val = {domain: (apply(x), y, g) for domain, (x, y, g) in raw_val.items()}
    target_split = load_domain_split(data_dir, eval_domain)
    target_X = np.load(data_dir / f"{eval_domain}_X.npy", mmap_mode="r")
    target_idx = np.asarray(target_split["test_idx"], dtype=np.int64)
    target_train_idx = np.asarray(target_split["train_idx"], dtype=np.int64)
    target_val_idx = np.asarray(target_split["val_idx"], dtype=np.int64)
    target_train_groups = np.asarray(target_split["row_group_id"][target_train_idx])
    target_val_groups = np.asarray(target_split["row_group_id"][target_val_idx])
    target_test_groups_full = np.asarray(target_split["row_group_id"][target_idx])
    overlap_audit[eval_domain] = {
        "train_val": int(len(np.intersect1d(np.unique(target_train_groups), np.unique(target_val_groups)))),
        "train_test": int(len(np.intersect1d(np.unique(target_train_groups), np.unique(target_test_groups_full)))),
        "val_test": int(len(np.intersect1d(np.unique(target_val_groups), np.unique(target_test_groups_full)))),
    }
    if max_target_rows is not None:
        target_rep = _representative_indices(target_test_groups_full, max_target_rows, 9300)
        target_idx = target_idx[target_rep]
    target_raw, target_names = transform_raw(np.asarray(target_X[target_idx]), names, representation)
    if target_names != selected_names:
        raise AssertionError("target representation names differ")
    split_hash = sha256_bytes("".join(split_hash_parts + [sha256_file(data_dir / "splits" / f"{eval_domain}_split.npz")]).encode())
    target_groups = np.asarray(target_split["row_group_id"][target_idx])
    preprocessing_hash = sha256_bytes(
        json.dumps(selected_names, separators=(",", ":")).encode()
        + np.ascontiguousarray(prep["lower"]).view(np.uint8).tobytes()
        + np.ascontiguousarray(prep["upper"]).view(np.uint8).tobytes()
        + np.ascontiguousarray(prep["median"]).view(np.uint8).tobytes()
        + np.ascontiguousarray(scaler.mean_).view(np.uint8).tobytes()
        + np.ascontiguousarray(scaler.scale_).view(np.uint8).tobytes()
    )
    return {
        "train_domains": list(train_domains),
        "eval_domain": eval_domain,
        "representation": representation,
        "raw_feature_names": names,
        "selected_feature_names": selected_names,
        "feature_count_raw": len(selected_names),
        "feature_count_input": int(next(iter(train.values()))[0].shape[1]),
        "train": train,
        "val": val,
        "target_test_features": apply(target_raw),
        "target_test_groups": target_groups,
        "group_overlap_audit": overlap_audit,
        "data_dir": str(data_dir.resolve()),
        "target_label_path": str((data_dir / f"{eval_domain}_y.npy").resolve()),
        "target_test_indices": target_idx,
        "split_hash": split_hash,
        "preprocessing_hash": preprocessing_hash,
        "code_sha256": code_fingerprint(),
        "config_sha256": config_fingerprint(),
    }


class DomainMLP(nn.Module):
    def __init__(self, in_dim: int, n_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, DOMAIN_MLP_HIDDEN[0]),
            nn.ReLU(),
            nn.Dropout(DOMAIN_MLP_DROPOUT),
            nn.Linear(DOMAIN_MLP_HIDDEN[0], DOMAIN_MLP_HIDDEN[1]),
            nn.ReLU(),
            nn.Dropout(DOMAIN_MLP_DROPOUT),
            nn.Linear(DOMAIN_MLP_HIDDEN[1], n_domains),
        )

    def forward(self, x):
        return self.net(x)


def _balanced_domain_rows(rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], task: str, seed: int):
    filtered = {}
    for domain, (X, y, g) in rows.items():
        if task == "all":
            mask = np.ones(len(y), dtype=bool)
        elif task == "benign":
            mask = y == 0
        elif task == "attack":
            mask = y == 1
        else:
            raise ValueError(task)
        filtered[domain] = (X[mask], y[mask], g[mask])
    n = min(len(value[0]) for value in filtered.values())
    if n == 0:
        raise ValueError(f"task {task} has no rows in all source domains")
    result = []
    for offset, domain in enumerate(rows):
        X, _, _ = filtered[domain]
        rng = np.random.default_rng(seed + offset)
        idx = np.arange(len(X)) if len(X) == n else np.sort(rng.choice(len(X), n, replace=False))
        result.append(X[idx])
    return np.concatenate(result), np.repeat(np.arange(len(rows)), n).astype(np.int64)


def _domain_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        out = {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y_true, probs, labels=np.arange(probs.shape[1]))),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=np.arange(probs.shape[1])).tolist(),
        }
    try:
        out["macro_ovr_auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_ovr_auroc"] = 0.5
    return out


def _domain_predict(model, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start:start + batch_size], dtype=np.float32)).to(device)
            out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _benchmark_inference(model, X: np.ndarray, device: str, warmup_repeats: int = 2,
                         measured_repeats: int = 5) -> dict:
    """Measure frozen-model inference only; no labels or selection is involved."""
    sample = np.asarray(X[:min(len(X), 8192)], dtype=np.float32)
    for _ in range(warmup_repeats):
        _ = predict_logits_batched(model, sample, device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    elapsed = []
    for _ in range(measured_repeats):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        _ = predict_logits_batched(model, X, device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
    median_sec = float(np.median(elapsed)) if elapsed else float("nan")
    return {
        "n_rows": int(len(X)),
        "warmup_repeats": int(warmup_repeats),
        "measured_repeats": int(measured_repeats),
        "latency_ms_per_1000_rows": float(median_sec * 1000.0 / max(len(X), 1) * 1000.0),
        "throughput_rows_per_sec": float(len(X) / median_sec) if median_sec > 0 else float("nan"),
        "measured_seconds": [float(x) for x in elapsed],
    }


def _fit_domain_mlp(X_train, y_train, X_val, y_val, seed: int, device: str):
    seed_everything(seed)
    model = DomainMLP(X_train.shape[1], int(y_train.max()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train)),
                        batch_size=DOMAIN_MLP_BATCH, shuffle=True, num_workers=0, drop_last=False)
    best_state = None
    best_key = None
    stale = 0
    history = []
    for epoch in range(1, DOMAIN_MLP_MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        probs = _domain_predict(model, X_val, device)
        metrics = _domain_metrics(y_val, probs)
        key = (metrics["macro_ovr_auroc"], metrics["balanced_accuracy"], -epoch)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= DOMAIN_MLP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_key": list(best_key), "epochs_run": len(history), "history": history}


def _permutation_importance(predict_fn, X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    if len(X) > PERMUTATION_MAX_ROWS:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), PERMUTATION_MAX_ROWS, replace=False))
        X = X[idx]; y = y[idx]
    base = _domain_metrics(y, predict_fn(X))["macro_ovr_auroc"]
    rng = np.random.default_rng(seed)
    values = []
    for j in range(X.shape[1]):
        perm = X.copy()
        perm[:, j] = perm[rng.permutation(len(perm)), j]
        score = _domain_metrics(y, predict_fn(perm))["macro_ovr_auroc"]
        values.append(float(base - score))
    return {"baseline_macro_ovr_auroc": float(base), "mean_decrease_by_input_index": values, "n_rows": int(len(X))}


def run_domain_coordinate(outer: str, representation: str, task: str, model_name: str, seed: int,
                          output_root: Path = OUTPUT_ROOT, data_dir: Path = DATA_DIR,
                          device: str = "cpu", max_domain_groups: int = MAX_DOMAIN_GROUPS,
                          smoke: bool = False, force: bool = False) -> dict:
    if outer not in DOMAINS or representation not in REPRESENTATIONS or task not in DOMAIN_TASKS or model_name not in DOMAIN_MODELS:
        raise ValueError("invalid domain coordinate")
    out_dir = output_root / "domain" / f"target_{outer}" / representation / task / model_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text())
        if existing.get("code_sha256") == code_fingerprint() and existing.get("config_sha256") == config_fingerprint():
            return existing
    started = time.time()
    source_domains = tuple(d for d in DOMAINS if d != outer)
    view = _preprocess_view(data_dir, source_domains, outer, representation,
                            max_train_rows=max_domain_groups,
                            max_val_rows=max_domain_groups if smoke else None,
                            max_target_rows=max_domain_groups,
                            domain_task=task)
    train_X, train_y = _balanced_domain_rows(view["train"], task, seed)
    val_rows = view["val"]
    if any(len(X) == 0 for X, _, _ in val_rows.values()):
        missing = [domain for domain, (X, _, _) in val_rows.items() if len(X) == 0]
        raise ValueError(f"task {task} has no source-validation group representatives for {missing}")
    val_X = np.concatenate([x for x, _, _ in val_rows.values()])
    val_y = np.repeat(np.arange(len(source_domains)), [len(x) for x, _, _ in val_rows.values()]).astype(np.int64)
    base_metrics = {}
    if model_name == "logreg":
        model = LogisticRegression(max_iter=300, multi_class="multinomial", class_weight="balanced", random_state=seed)
        model.fit(train_X, train_y)
        predict_fn = lambda X: model.predict_proba(X)
        train_meta = {"solver": "lbfgs", "max_iter": 300, "class_weight": "balanced"}
    else:
        model, train_meta = _fit_domain_mlp(train_X, train_y, val_X, val_y, seed, device)
        predict_fn = lambda X: _domain_predict(model, X, device)
    val_probs = predict_fn(val_X)
    base_metrics = _domain_metrics(val_y, val_probs)
    target_probs = predict_fn(view["target_test_features"])
    importance = _permutation_importance(predict_fn, val_X, val_y, seed + 30000)
    target_counts = np.bincount(np.argmax(target_probs, axis=1), minlength=len(source_domains)).tolist()
    result = {
        "status": "success",
        "protocol_version": PROTOCOL_VERSION,
        "kind": "domain",
        "outer_target": outer,
        "source_domains": list(source_domains),
        "representation": representation,
        "task": task,
        "model": model_name,
        "seed": int(seed),
        "feature_names": view["selected_feature_names"],
        "feature_count_input": view["feature_count_input"],
        "source_train_group_representatives": int(len(train_X)),
        "source_validation_rows": int(len(val_X)),
        "source_validation_metrics": base_metrics,
        "permutation_importance": importance,
        "target_posthoc_unlabeled_prediction_counts": target_counts,
        "target_labels_used_for_training_or_selection": False,
        "target_labels_read": False,
        "target_posthoc_is_selection_free": True,
        "group_overlap_audit": view["group_overlap_audit"],
        "split_hash": view["split_hash"],
        "preprocessing_hash": view["preprocessing_hash"],
        "code_sha256": code_fingerprint(),
        "config_sha256": config_fingerprint(),
        "training": train_meta,
        "runtime": {"device": str(device), "torch": torch.__version__},
        "elapsed_sec": round(time.time() - started, 3),
    }
    if model_name == "mlp":
        torch.save(model.state_dict(), out_dir / "model.pt")
        result["model_sha256"] = sha256_file(out_dir / "model.pt")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_model_coordinate(outer: str, representation: str, seed: int,
                         output_root: Path = OUTPUT_ROOT, data_dir: Path = DATA_DIR,
                         device: str = "cuda", max_epochs: int = ERM_MAX_EPOCHS,
                         smoke: bool = False, force: bool = False) -> dict:
    if outer not in DOMAINS or representation not in REPRESENTATIONS:
        raise ValueError("invalid model coordinate")
    out_dir = output_root / "model" / f"target_{outer}" / representation / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text())
        if existing.get("code_sha256") == code_fingerprint() and existing.get("config_sha256") == config_fingerprint():
            return existing
    started = time.time()
    source_domains = tuple(d for d in DOMAINS if d != outer)
    view = _preprocess_view(data_dir, source_domains, outer, representation,
                            max_train_rows=10000 if smoke else None,
                            max_val_rows=5000 if smoke else None,
                            max_target_rows=5000 if smoke else None,
                            preserve_binary_classes=smoke)
    model, train_meta = train_source_erm(view, seed, device, max_epochs=max_epochs, patience=ERM_PATIENCE)
    inference = _benchmark_inference(model, view["target_test_features"], device)
    metrics, predictions = evaluate_model(model, view, device)
    np.savez_compressed(out_dir / "predictions.npz", **predictions)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, out_dir / "model.pt")
    result = {
        "status": "success",
        "protocol_version": PROTOCOL_VERSION,
        "kind": "model",
        "outer_target": outer,
        "source_domains": list(source_domains),
        "representation": representation,
        "seed": int(seed),
        "max_epochs": int(max_epochs),
        "feature_names": view["selected_feature_names"],
        "feature_count_input": view["feature_count_input"],
        "target_labels_used_for_training_or_selection": False,
        "target_labels_read": True,
        "target_labels_read_after_model_and_threshold_freeze": True,
        "group_overlap_audit": view["group_overlap_audit"],
        "split_hash": view["split_hash"],
        "preprocessing_hash": view["preprocessing_hash"],
        "train": train_meta,
        "metrics": metrics,
        "inference": inference,
        "runtime": {"device": str(device), "torch": torch.__version__,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else 0},
        "model_sha256": sha256_file(out_dir / "model.pt"),
        "predictions_sha256": sha256_file(out_dir / "predictions.npz"),
        "code_sha256": code_fingerprint(),
        "config_sha256": config_fingerprint(),
        "elapsed_sec": round(time.time() - started, 3),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_protocol_manifest(output_root: Path, data_dir: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "protocol_manifest.json"
    current_code = code_fingerprint()
    current_config = config_fingerprint()
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("code_sha256") == current_code and existing.get("config_sha256") == current_config:
            return existing
    names_path = data_dir / "feature_names.txt"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "domains": list(DOMAINS),
        "representations": list(REPRESENTATIONS),
        "domain_tasks": list(DOMAIN_TASKS),
        "domain_models": list(DOMAIN_MODELS),
        "seeds": list(SEEDS),
        "feature_schema_sha256": sha256_file(names_path),
        "feature_names": feature_names(data_dir),
        "feature_count_raw": RAW_FEATURE_COUNT,
        "relational_feature_names": [name for name, _, _ in RELATIONAL_SPECS],
        "split_hashes": {domain: sha256_file(data_dir / "splits" / f"{domain}_split.npz") for domain in DOMAINS},
        "data_array_sha256": {
            domain: {
                "X": sha256_file(data_dir / f"{domain}_X.npy"),
                "y": sha256_file(data_dir / f"{domain}_y.npy"),
            } for domain in DOMAINS
        },
        "external_data_allowed": False,
        "target_labels_used_for_training_or_selection": False,
        "code_sha256": current_code,
        "config_sha256": current_config,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("domain", "model"), required=True)
    parser.add_argument("--outer", choices=DOMAINS, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--task", choices=DOMAIN_TASKS, default="all")
    parser.add_argument("--model", choices=DOMAIN_MODELS, default="logreg")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=ERM_MAX_EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute this coordinate even if result.json exists")
    args = parser.parse_args()
    if args.write_manifest:
        print(json.dumps(write_protocol_manifest(args.output_root, args.data_dir), indent=2, ensure_ascii=False))
    if args.kind == "domain":
        result = run_domain_coordinate(args.outer, args.representation, args.task, args.model, args.seed,
                                       args.output_root, args.data_dir, args.device, smoke=args.smoke, force=args.force)
    else:
        result = run_model_coordinate(args.outer, args.representation, args.seed,
                                      args.output_root, args.data_dir, args.device,
                                      max_epochs=args.max_epochs, smoke=args.smoke, force=args.force)
    print(json.dumps({"status": result["status"], "kind": result["kind"],
                      "outer": result["outer_target"], "representation": result["representation"],
                      "seed": result["seed"], "elapsed_sec": result["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
