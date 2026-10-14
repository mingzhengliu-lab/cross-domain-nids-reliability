"""NumPy-only split, duplicate-group and source preprocessing helpers.

The experiment loader re-exports these same functions. CPU Q/control runs and
artifact checks therefore do not need the neural training runtime.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

SPLIT_DIRNAME = "splits"
FEATURE_PREPROCESS_POLICY = "source_train_quantile_cap_scale_invalid_indicators_v1"
INVALID_SENTINEL_ABS = 1e9
CAP_LOWER_Q = 0.001
CAP_UPPER_Q = 0.999

def exact_row_group_ids(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact byte-level row group ids and group counts.

    The processed NFv2 arrays are numeric feature matrices. Treating each row as
    a fixed-width byte record lets us group exact duplicate feature rows without
    a lossy hash collision risk.
    """
    Xc = np.ascontiguousarray(X)
    row_dtype = np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))
    row_view = Xc.view(row_dtype).reshape(-1)
    _, inverse, counts = np.unique(row_view, return_inverse=True, return_counts=True)
    return inverse.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)

def unique_row_weights(X: np.ndarray) -> np.ndarray:
    """Weight each exact duplicate group as one evaluation unit."""
    group_ids, counts = exact_row_group_ids(X)
    return (1.0 / counts[group_ids]).astype(np.float64, copy=False)

def load_domain_split(data_dir: str | Path, domain: str) -> dict[str, np.ndarray]:
    path = Path(data_dir) / SPLIT_DIRNAME / f"{domain}_split.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing duplicate-aware split file: {path}. "
            "Restore the frozen DATA/nfv2_processed/splits keys before training."
        )
    data = np.load(path, allow_pickle=False)
    return {
        "train_idx": data["train_idx"].astype(np.int64, copy=False),
        "val_idx": data["val_idx"].astype(np.int64, copy=False),
        "test_idx": data["test_idx"].astype(np.int64, copy=False),
        "row_group_id": data["row_group_id"].astype(np.int64, copy=False),
    }

def _invalid_mask(X: np.ndarray) -> np.ndarray:
    return (~np.isfinite(X)) | (np.abs(X) >= INVALID_SENTINEL_ABS)

def fit_source_train_preprocessor(X_train: np.ndarray, feature_names: Optional[list[str]] = None) -> dict:
    """Fit finite-value imputation/capping stats on source train only."""
    X = np.asarray(X_train)
    n_features = X.shape[1]
    lower = np.zeros(n_features, dtype=np.float32)
    upper = np.zeros(n_features, dtype=np.float32)
    median = np.zeros(n_features, dtype=np.float32)
    invalid_rates = np.zeros(n_features, dtype=np.float32)
    invalid = _invalid_mask(X)

    for j in range(n_features):
        col = X[:, j]
        bad = invalid[:, j]
        invalid_rates[j] = float(bad.mean())
        finite = col[~bad]
        if finite.size == 0:
            lower[j] = upper[j] = median[j] = 0.0
            continue
        lower[j] = float(np.quantile(finite, CAP_LOWER_Q))
        upper[j] = float(np.quantile(finite, CAP_UPPER_Q))
        if lower[j] > upper[j]:
            lower[j], upper[j] = upper[j], lower[j]
        median[j] = float(np.median(finite))

    return {
        "policy": FEATURE_PREPROCESS_POLICY,
        "lower": lower,
        "upper": upper,
        "median": median,
        "invalid_rates": invalid_rates,
        "feature_names": feature_names,
    }

def apply_source_train_preprocessor(X: np.ndarray, prep: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply source-train fitted invalid handling and return (features, indicators)."""
    Xf = np.asarray(X, dtype=np.float32).copy()
    invalid = _invalid_mask(Xf)
    lower = prep["lower"]
    upper = prep["upper"]
    median = prep["median"]
    for j in range(Xf.shape[1]):
        bad = invalid[:, j]
        if bad.any():
            Xf[bad, j] = median[j]
        np.clip(Xf[:, j], lower[j], upper[j], out=Xf[:, j])
    return Xf, invalid.astype(np.float32, copy=False)
