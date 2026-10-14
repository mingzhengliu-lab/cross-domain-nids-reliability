"""Model-evaluation and hyperparameter-selection helpers.

The main protocol now treats AUROC/AUPRC/low-FPR TPR as primary deployment
signals and keeps fixed-threshold F1 as a diagnostic metric. Unique-row metrics
can be weighted with raw duplicate-group weights supplied by the caller.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .datasets import unique_row_weights


# ============================================================
# ============================================================

@torch.no_grad()
def predict_logits_batched(algorithm, X: np.ndarray, batch_size: int = 4096,
                           device: str = 'cuda') -> np.ndarray:
    """
    Return logits in batches without changing the evaluation protocol.

    This is used by calibration and artifact backfills so they do not need to
    duplicate inference loops.
    """
    batch_size = int(os.environ.get("NIDS_EVAL_BATCH", batch_size))
    algorithm.eval()

    logits_list = []
    for i in range(0, len(X), batch_size):
        x_batch = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        logits = algorithm.predict(x_batch)
        logits_list.append(logits.cpu().numpy())

    return np.concatenate(logits_list, axis=0)


@torch.no_grad()
def predict_batched(algorithm, X: np.ndarray, batch_size: int = 4096,
                    device: str = 'cuda') -> tuple:
    """
    Batched prediction, avoiding evaluation OOM.

    Returns:
        (preds, probs)
        preds:  (N,) int64
        probs:  (N,) float32 for class 1
    """
    logits = predict_logits_batched(algorithm, X, batch_size=batch_size, device=device)
    logits_t = torch.from_numpy(logits)
    probs = F.softmax(logits_t, dim=1)[:, 1].numpy()
    preds = logits_t.argmax(dim=1).numpy()
    return preds, probs


def _safe_auroc(y: np.ndarray, scores: np.ndarray, sample_weight=None) -> float:
    try:
        return float(roc_auc_score(y, scores, sample_weight=sample_weight))
    except ValueError:
        return float('nan')


def _safe_auprc(y: np.ndarray, scores: np.ndarray, sample_weight=None) -> float:
    try:
        return float(average_precision_score(y, scores, sample_weight=sample_weight))
    except ValueError:
        return float('nan')


def _tpr_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_target: float,
                sample_weight=None) -> float:
    try:
        fpr, tpr, _ = roc_curve(y, scores, sample_weight=sample_weight)
    except ValueError:
        return float('nan')
    if fpr_target <= fpr[0]:
        return float(tpr[0])
    if fpr_target >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(fpr_target, fpr, tpr))


def select_threshold_at_max_fpr(
    y_reference: np.ndarray,
    scores_reference: np.ndarray,
    max_fpr: float,
) -> dict:
    """Select a conservative fixed threshold from reference negatives only.

    Predictions use ``score >= threshold``. If ties at the boundary would exceed
    ``max_fpr``, the threshold is advanced to the next representable float. No
    target-evaluation labels or ROC interpolation are used.
    """
    y_reference = np.asarray(y_reference, dtype=np.int64)
    scores_reference = np.asarray(scores_reference, dtype=np.float64)
    max_fpr = float(max_fpr)
    if y_reference.ndim != 1 or scores_reference.ndim != 1:
        raise ValueError("reference labels and scores must be one-dimensional")
    if len(y_reference) != len(scores_reference):
        raise ValueError("reference labels and scores must have equal length")
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in [0,1], got {max_fpr}")
    if not np.isfinite(scores_reference).all():
        raise ValueError("reference scores contain NaN or Inf")

    negative_scores = scores_reference[y_reference == 0]
    n_negative = int(len(negative_scores))
    if n_negative == 0:
        raise ValueError("cannot select an FPR threshold without reference negatives")

    allowed_fp = int(np.floor(max_fpr * n_negative + 1e-12))
    ordered = np.sort(negative_scores)[::-1]
    if allowed_fp <= 0:
        threshold = np.nextafter(ordered[0], np.inf)
    elif allowed_fp >= n_negative:
        threshold = ordered[-1]
    else:
        boundary = ordered[allowed_fp - 1]
        threshold = boundary
        if int((negative_scores >= threshold).sum()) > allowed_fp:
            threshold = np.nextafter(boundary, np.inf)

    achieved_reference_fpr = float((negative_scores >= threshold).mean())
    min_observable_fpr = 1.0 / n_negative
    return {
        "threshold": float(threshold),
        "max_fpr_requested": max_fpr,
        "reference_fpr": achieved_reference_fpr,
        "n_reference": int(len(y_reference)),
        "n_reference_negative": n_negative,
        "allowed_false_positives": allowed_fp,
        "min_observable_fpr": float(min_observable_fpr),
        "underresolved": bool(0.0 < max_fpr < min_observable_fpr),
        "selection_policy": "reference_negative_empirical_conservative_v1",
    }


def metrics_at_fixed_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate operational metrics at a preselected immutable threshold."""
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    threshold = float(threshold)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")

    pred = scores >= threshold
    positive = y == 1
    negative = ~positive
    tp = int((pred & positive).sum())
    fp = int((pred & negative).sum())
    tn = int((~pred & negative).sum())
    fn = int((~pred & positive).sum())
    n_positive = tp + fn
    n_negative = fp + tn
    n = len(y)

    tpr = float(tp / n_positive) if n_positive else float("nan")
    fpr = float(fp / n_negative) if n_negative else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    alert_rate = float((tp + fp) / n) if n else float("nan")
    return {
        "threshold": threshold,
        "n_samples": int(n),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "recall": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "alert_rate": alert_rate,
        "alerts_per_million_flows": float(alert_rate * 1_000_000),
        "false_alerts_per_million_benign": float(fpr * 1_000_000),
        "evaluation_policy": "fixed_reference_threshold_v1",
    }


def _classification_metrics(y: np.ndarray, preds: np.ndarray, probs: np.ndarray,
                            sample_weight=None) -> dict:
    return {
        'attack_f1': float(f1_score(
            y, preds, pos_label=1, zero_division=0, sample_weight=sample_weight
        )),
        'auroc': _safe_auroc(y, probs, sample_weight=sample_weight),
        'auprc': _safe_auprc(y, probs, sample_weight=sample_weight),
        'precision': float(precision_score(
            y, preds, pos_label=1, zero_division=0, sample_weight=sample_weight
        )),
        'recall': float(recall_score(
            y, preds, pos_label=1, zero_division=0, sample_weight=sample_weight
        )),
        'tpr_at_fpr_1pct': _tpr_at_fpr(y, probs, 0.01, sample_weight=sample_weight),
        'tpr_at_fpr_0p1pct': _tpr_at_fpr(y, probs, 0.001, sample_weight=sample_weight),
        'tpr_at_fpr_0p01pct': _tpr_at_fpr(y, probs, 0.0001, sample_weight=sample_weight),
    }


def evaluate_attack_f1(algorithm, X: np.ndarray, y: np.ndarray,
                       batch_size: int = 4096, device: str = 'cuda') -> float:
    """Compute attack-class F1 (positive label = 1) with argmax predictions."""
    preds, _ = predict_batched(algorithm, X, batch_size, device)
    return float(f1_score(y, preds, pos_label=1, zero_division=0))


def evaluate_auroc(algorithm, X: np.ndarray, y: np.ndarray,
                   batch_size: int = 4096, device: str = 'cuda') -> float:
    """Compute attack-vs-benign AUROC for ``(X, y)``, batched."""
    _, probs = predict_batched(algorithm, X, batch_size, device)
    return _safe_auroc(y, probs)


def evaluate_full(algorithm, X: np.ndarray, y: np.ndarray,
                  batch_size: int = 4096, device: str = 'cuda',
                  unique_weights: np.ndarray | None = None) -> dict:
    """Run the full binary classification metric bundle on ``(X, y)``.

    Returns a dict with attack_f1, auroc, auprc, precision, recall,
    attack_ratio, n_samples, and duplicate-aware ``unique_*`` variants
    when unique-row weights can be derived.
    """
    preds, probs = predict_batched(algorithm, X, batch_size, device)
    if unique_weights is None:
        try:
            unique_weights = unique_row_weights(X)
        except Exception:
            unique_weights = None

    return evaluate_score_arrays(
        y,
        probs,
        preds=preds,
        unique_weights=unique_weights,
    )


def evaluate_score_arrays(y: np.ndarray, scores: np.ndarray,
                          preds: np.ndarray | None = None,
                          unique_weights: np.ndarray | None = None) -> dict:
    """Evaluate persisted attack scores with the same bundle as model inference."""
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if preds is None:
        preds = (scores >= 0.5).astype(np.int64)
    else:
        preds = np.asarray(preds, dtype=np.int64)
        if preds.shape != y.shape:
            raise ValueError("preds must have the same shape as y")

    out = _classification_metrics(y, preds, scores)
    out['target_oracle_roc'] = {
        'tpr_at_fpr_1pct': out['tpr_at_fpr_1pct'],
        'tpr_at_fpr_0p1pct': out['tpr_at_fpr_0p1pct'],
        'tpr_at_fpr_0p01pct': out['tpr_at_fpr_0p01pct'],
        'uses_evaluation_labels': True,
        'deployment_threshold_metric': False,
    }
    out.update({
        'attack_ratio': float((y == 1).mean()),
        'n_samples': int(len(y)),
    })

    try:
        if unique_weights is None:
            raise ValueError("unique weights were not supplied for persisted scores")
        weights = unique_weights
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(y):
            raise ValueError("unique_weights length does not match y")
        unique = _classification_metrics(y, preds, scores, sample_weight=weights)
        for key, value in unique.items():
            out[f'unique_{key}'] = value
        out['n_unique_feature_rows'] = float(weights.sum())
    except Exception:
        pass

    return out


# ============================================================
# ============================================================

def evaluate_source_val_macro_avg(algorithm, val_per_domain: dict,
                                  batch_size: int = 4096,
                                  device: str = 'cuda',
                                  unique_weights_per_domain: dict | None = None) -> dict:
    """Return macro-averaged source-validation metrics across source domains."""
    per_domain = {}
    unique_weights_per_domain = unique_weights_per_domain or {}
    for dom, (X, y) in val_per_domain.items():
        per_domain[dom] = evaluate_full(
            algorithm,
            X,
            y,
            batch_size,
            device,
            unique_weights=unique_weights_per_domain.get(dom),
        )

    def _macro(key: str) -> float:
        vals = [m.get(key, float('nan')) for m in per_domain.values()]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    return {
        'val_attack_f1_macro': _macro('attack_f1'),
        'val_auroc_macro': _macro('auroc'),
        'val_auprc_macro': _macro('auprc'),
        'val_tpr_at_fpr_1pct_macro': _macro('tpr_at_fpr_1pct'),
        'val_unique_attack_f1_macro': _macro('unique_attack_f1'),
        'val_unique_auroc_macro': _macro('unique_auroc'),
        'val_attack_f1_per_domain': {dom: m['attack_f1'] for dom, m in per_domain.items()},
        'val_metrics_per_domain': per_domain,
    }


# ============================================================
# ============================================================

def select_best_hparam(search_results: list, metric_key: str = 'val_auroc_macro') -> dict:
    """Return the search record with the largest requested validation metric."""
    if not search_results:
        raise ValueError("search_results is empty")

    best = max(search_results, key=lambda r: r.get(metric_key, -float('inf')))
    return best
