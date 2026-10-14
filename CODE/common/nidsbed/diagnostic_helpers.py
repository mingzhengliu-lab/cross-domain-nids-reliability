"""Shared functions used by the paper's NFv2 diagnostics and low-budget runs.

Function bodies retain the original weighted-threshold and source-ERM rules.
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from nidsbed.algorithms.erm import ERM
from nidsbed.calibration import binary_ece
from nidsbed.datasets import LODODataset, GPUResidentBatchIterator, build_train_loader
from nidsbed.lib.misc import get_class_weights, seed_everything
from nidsbed.model_selection import (evaluate_score_arrays, metrics_at_fixed_threshold,
    predict_logits_batched, select_threshold_at_max_fpr)

BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 6
HYPERPARAMS = {'lr': 1e-4, 'batch_size': BATCH_SIZE, 'weight_decay': 1e-5,
              'dropout': 0.3, 'hidden': 256, 'feat_dim': 64}

def select_weighted_threshold_at_max_fpr(y_reference: np.ndarray,
                                         scores_reference: np.ndarray,
                                         weights: np.ndarray,
                                         max_fpr: float) -> dict:
    y_reference = np.asarray(y_reference, dtype=np.int64)
    scores_reference = np.asarray(scores_reference, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (len(y_reference) == len(scores_reference) == len(weights)):
        raise ValueError("weighted threshold arrays must have equal length")
    if not np.isfinite(scores_reference).all() or not np.isfinite(weights).all():
        raise ValueError("weighted threshold inputs must be finite")
    negative = y_reference == 0
    negative_scores = scores_reference[negative]
    negative_weights = weights[negative]
    if not len(negative_scores) or negative_weights.sum() <= 0:
        raise ValueError("weighted threshold requires reference negatives")
    order = np.argsort(-negative_scores, kind="mergesort")
    ordered_scores = negative_scores[order]
    ordered_weights = negative_weights[order]
    total_weight = float(ordered_weights.sum())
    allowed_weight = float(max_fpr) * total_weight
    threshold = np.nextafter(float(ordered_scores[0]), np.inf)
    used_weight = 0.0
    index = 0
    while index < len(ordered_scores):
        score = ordered_scores[index]
        end = index + 1
        while end < len(ordered_scores) and ordered_scores[end] == score:
            end += 1
        tie_weight = float(ordered_weights[index:end].sum())
        if used_weight + tie_weight > allowed_weight + 1e-15:
            threshold = np.nextafter(float(score), np.inf)
            break
        used_weight += tie_weight
        threshold = float(score)
        index = end
    achieved = float(
        negative_weights[negative_scores >= threshold].sum() / total_weight
    )
    return {
        "threshold": float(threshold),
        "max_fpr_requested": float(max_fpr),
        "reference_fpr": achieved,
        "reference_negative_weight": total_weight,
        "allowed_false_positive_weight": allowed_weight,
        "selection_policy": "reference_negative_group_weighted_conservative_v1",
    }


def weighted_fixed_threshold(y, probability, threshold, weights):
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    pred = probability >= float(threshold)
    weights = np.asarray(weights, dtype=np.float64)
    positive = y == 1
    negative = ~positive
    tp = float(weights[pred & positive].sum())
    fp = float(weights[pred & negative].sum())
    tn = float(weights[(~pred) & negative].sum())
    fn = float(weights[(~pred) & positive].sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "tpr": float(tpr),
        "recall": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "alerts_per_million_flows": float((tp + fp) / weights.sum() * 1_000_000),
        "false_alerts_per_million_benign": float(fpr * 1_000_000),
    }


def group_weights(group_ids: np.ndarray) -> np.ndarray:
    """Return row weights giving every exact duplicate group unit mass."""
    _, inverse, counts = np.unique(np.asarray(group_ids), return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float64, copy=False)


def combine_domain_group_ids(groups_by_domain: list[np.ndarray]) -> np.ndarray:
    """Encode domain-local group IDs in one collision-free integer namespace."""
    encoded = []
    offset = 0
    for groups in groups_by_domain:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        encoded.append(inverse.astype(np.int64, copy=False) + offset)
        offset += int(inverse.max()) + 1 if len(inverse) else 0
    return np.concatenate(encoded) if encoded else np.empty(0, dtype=np.int64)


def _source_val_metrics(model: ERM, view: dict, device: str) -> dict:
    rows, groups, tprs = [], [], []
    per_domain = {}
    for domain in view["train_domains"]:
        X, y, g = view["val"][domain]
        logits = predict_logits_batched(model, X, device=device)
        score = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
        w = group_weights(g)
        metric = evaluate_score_arrays(y, score, unique_weights=w)
        per_domain[domain] = metric
        rows.append(metric["auroc"]); groups.append(metric.get("unique_auroc", metric["auroc"]))
        tprs.append(metric["tpr_at_fpr_1pct"])
    return {
        "per_domain": per_domain,
        "row_auroc": float(np.mean(rows)),
        "group_auroc": float(np.mean(groups)),
        "utility": float(min(np.mean(rows), np.mean(groups))),
        "low_fpr_tpr": float(np.mean(tprs)),
    }


def train_source_erm(view: dict, seed: int, device: str, max_epochs: int = MAX_EPOCHS,
                     patience: int = PATIENCE):
    seed_everything(seed)
    datasets = []
    weights = {}
    for label, domain in enumerate(view["train_domains"]):
        X, y, _ = view["train"][domain]
        datasets.append(LODODataset(X, y, domain_label=label))
        weights[label] = get_class_weights(y, clip_max=20.0, device=device)
    hparams = dict(HYPERPARAMS)
    model = ERM(X.shape[1], 2, hparams, weights, device=device)
    if str(device).startswith("cuda"):
        loader = GPUResidentBatchIterator(datasets, BATCH_SIZE, False, device, seed)
    else:
        loader = build_train_loader(datasets, BATCH_SIZE, False, 0, seed)
    best_key = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            info = model.update(batch)
            losses.append(float(info["loss_total"]))
        val = _source_val_metrics(model, view, device)
        key = (val["utility"], val["low_fpr_tpr"], val["row_auroc"] + val["group_auroc"], -epoch)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **{k: val[k] for k in ("row_auroc", "group_auroc", "utility", "low_fpr_tpr")}})
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_key": list(best_key), "epochs_run": len(history), "history": history, "hparams": hparams}


def _scores(model: ERM, X: np.ndarray, device: str) -> np.ndarray:
    logits = predict_logits_batched(model, X, device=device)
    logits = torch.from_numpy(logits)
    return torch.softmax(logits, dim=1)[:, 1].numpy().astype(np.float64)


def _pauc(y: np.ndarray, score: np.ndarray, weights: np.ndarray | None = None) -> float:
    try:
        return float(roc_auc_score(y, score, sample_weight=weights, max_fpr=0.01))
    except ValueError:
        return float("nan")


def evaluate_model(model: ERM, view: dict, device: str) -> tuple[dict, dict]:
    source_scores, source_y, source_group_parts = [], [], []
    for domain in view["train_domains"]:
        X, y, g = view["val"][domain]
        source_scores.append(_scores(model, X, device)); source_y.append(y); source_group_parts.append(g)
    source_score = np.concatenate(source_scores); source_y = np.concatenate(source_y)
    source_groups = combine_domain_group_ids(source_group_parts)
    source_weights = group_weights(source_groups)
    row_threshold = select_threshold_at_max_fpr(source_y, source_score, 0.01)
    group_threshold = select_weighted_threshold_at_max_fpr(source_y, source_score, source_weights, 0.01)
    # Evaluation labels are opened only after the model and both source-only
    # operational thresholds are frozen.
    target_y_all = np.load(view["target_label_path"], mmap_mode="r")
    target_y = np.asarray(target_y_all[view["target_test_indices"]], dtype=np.int64)
    target_groups = view["target_test_groups"]
    target_score = _scores(model, view["target_test_features"], device)
    target_weights = group_weights(target_groups)
    ranking = evaluate_score_arrays(target_y, target_score, unique_weights=target_weights)
    ranking["partial_auc_1pct"] = _pauc(target_y, target_score)
    ranking["unique_partial_auc_1pct"] = _pauc(target_y, target_score, target_weights)
    row_operational = metrics_at_fixed_threshold(target_y, target_score, row_threshold["threshold"])
    group_operational = weighted_fixed_threshold(target_y, target_score, group_threshold["threshold"], target_weights)
    ranking["source_threshold_row"] = row_operational
    ranking["source_threshold_group"] = group_operational
    ranking["source_threshold_recall_row"] = float(row_operational["recall"])
    ranking["source_threshold_recall_group"] = float(group_operational["recall"])
    ranking["source_threshold_false_alerts_m_row"] = float(row_operational["false_alerts_per_million_benign"])
    ranking["source_threshold_false_alerts_m_group"] = float(group_operational["false_alerts_per_million_benign"])
    ranking["brier_row"] = float(np.mean((target_score - target_y) ** 2))
    ranking["brier_group"] = float(np.average((target_score - target_y) ** 2, weights=target_weights))
    ranking["ece_row"] = float(binary_ece(target_score, target_y))
    ranking["utility"] = float(min(ranking["auroc"], ranking.get("unique_auroc", ranking["auroc"])))
    predictions = {
        "target_scores": target_score.astype(np.float32),
        "target_labels": target_y.astype(np.int64),
        "target_group_ids": target_groups.astype(np.int64),
        "source_val_scores": source_score.astype(np.float32),
        "source_val_labels": source_y.astype(np.int64),
        "source_val_group_ids": source_groups.astype(np.int64),
    }
    return ranking, predictions
