"""Post-hoc probability calibration helpers.

The benchmark trains with source-domain class weighting, so raw softmax scores
are useful ranking scores but should not be treated as calibrated posteriors.
Temperature scaling on held-out source validation data gives downstream
calibration / energy / threshold-based diagnostics a well-defined input.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-7


def _as_numpy_1d(values) -> np.ndarray:
    arr = np.asarray(values)
    return arr.reshape(-1)


def nll_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    labels = _as_numpy_1d(labels).astype(np.int64)
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    with torch.no_grad():
        loss = F.cross_entropy(logits_t / float(temperature), labels_t)
    return float(loss.item())


def scores_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    with torch.no_grad():
        scores = F.softmax(logits_t / float(temperature), dim=1)[:, 1]
    return scores.cpu().numpy().astype(np.float32)


def energy_scores_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free-energy score: -T * logsumexp(logits / T, dim=1).

    This matches the OOD energy formulation (Liu et al. 2020). The sign here
    follows the score_fusion convention so a *higher* value means *less*
    confident (more OOD-like). It is not monotone in softmax P(y=1) and so
    carries information complementary to the softmax score under domain shift.
    """
    logits = np.asarray(logits, dtype=np.float64)
    T = float(temperature)
    scaled = logits / T
    max_per_row = np.max(scaled, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(scaled - max_per_row).sum(axis=1)) + max_per_row[:, 0]
    return (-T * logsumexp).astype(np.float32)


def binary_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    scores = np.clip(_as_numpy_1d(scores).astype(float), EPS, 1.0 - EPS)
    labels = _as_numpy_1d(labels).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        conf = float(scores[mask].mean())
        acc = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return float(ece)


def fit_temperature_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 80,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
) -> tuple[float, dict]:
    """Fit one scalar temperature by source-validation NLL.

    Returns:
        (temperature, metadata)
    """
    labels_np = _as_numpy_1d(labels).astype(np.int64)
    logits_np = np.asarray(logits, dtype=np.float32)
    if logits_np.ndim != 2 or logits_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            f"logits/labels shape mismatch: logits={logits_np.shape}, labels={labels_np.shape}"
        )

    unique = np.unique(labels_np)
    if len(unique) < 2:
        raw_scores = scores_from_logits(logits_np, temperature=1.0)
        meta = {
            "method": "temperature_scaling",
            "status": "skipped_single_class_source_val",
            "temperature": 1.0,
            "n_source_val": int(labels_np.shape[0]),
            "source_val_attack_rate": float(labels_np.mean()) if labels_np.size else float("nan"),
            "source_val_nll_raw": nll_from_logits(logits_np, labels_np, temperature=1.0),
            "source_val_nll_calibrated": nll_from_logits(logits_np, labels_np, temperature=1.0),
            "source_val_ece_raw": binary_ece(raw_scores, labels_np),
            "source_val_ece_calibrated": binary_ece(raw_scores, labels_np),
        }
        return 1.0, meta

    logits_t = torch.as_tensor(logits_np, dtype=torch.float32)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min_temperature, max_temperature)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature = float(torch.exp(log_temperature).clamp(min_temperature, max_temperature).item())

    if not math.isfinite(temperature):
        temperature = 1.0

    raw_scores = scores_from_logits(logits_np, temperature=1.0)
    calibrated_scores = scores_from_logits(logits_np, temperature=temperature)
    meta = {
        "method": "temperature_scaling",
        "status": "ok",
        "temperature": temperature,
        "n_source_val": int(labels_np.shape[0]),
        "source_val_attack_rate": float(labels_np.mean()),
        "source_val_nll_raw": nll_from_logits(logits_np, labels_np, temperature=1.0),
        "source_val_nll_calibrated": nll_from_logits(logits_np, labels_np, temperature=temperature),
        "source_val_ece_raw": binary_ece(raw_scores, labels_np),
        "source_val_ece_calibrated": binary_ece(calibrated_scores, labels_np),
    }
    return temperature, meta
