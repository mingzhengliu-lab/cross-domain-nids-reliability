"""Rigorous group-aware target-support adaptation from ERM checkpoints.

The runner keeps one preprocessed target loader alive while iterating over N and
adaptation modes. Support representatives come only from the persisted target
train split; final evaluation always uses the fixed, group-disjoint target test
split. Existing legacy outputs are never read or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

import numpy as np
import torch

from nidsbed.adaptation import (
    ADAPTATION_PROTOCOL_VERSION,
    ALL_ADAPTATION_MODES,
    DIAGNOSTIC_ONLY_MODES,
    LPFT_L2SP_COEFFICIENT,
    LPFT_MODES,
    adaptation_train_step,
    build_adaptation_optimizer,
    capture_backbone_reference,
    configure_adaptation_mode,
    configure_lpft_finetune,
    model_state_hashes,
    recalibrate_batch_norm,
    select_label_matched_source_replay,
)
from nidsbed.calibration import scores_from_logits
from nidsbed.datasets import (
    FEW_SHOT_POLICY_GROUP_NESTED,
    FEW_SHOT_POLICY_GROUP_RANDOM_NESTED,
    GPUResidentBatchIterator,
    LODODataLoader,
    LODODataset,
    load_domain_split,
    unique_row_weights,
)
from nidsbed.lib.misc import get_class_weights, seed_everything
from nidsbed.model_selection import (
    evaluate_full,
    metrics_at_fixed_threshold,
    predict_logits_batched,
    select_threshold_at_max_fpr,
)
from CODE.experiment_stage3.train import _build_algorithm_from_checkpoint_payload, get_algorithm_class


TARGETS = ["unsw", "cic", "bot", "ton"]
DEFAULT_NS = [50, 100, 500, 1000]
DEFAULT_MODES = ["full_ft", "target_only"]
SUPPORT_POLICIES = ("balanced", "random_group")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "RESULTS" / "low_label" / "revision_fewshot_v5"
DEFAULT_CHECKPOINT_TEMPLATE = (
    PROJECT_ROOT
    / "RESULTS"
    / "stage3_lodo"
    / "stage3"
    / "erm_{target}"
    / "final"
    / "seed{seed}_artifacts"
    / "checkpoint.pt"
)
FPR_BUDGETS = [0.01, 0.001, 0.0001]
SELECTION_ATOL = 1e-12
LPFT_PROBE_EPOCHS = 5
REPLAY_NEIGHBORS_PER_TARGET_PER_SOURCE = 2
REPLAY_TARGET_REPEAT_FACTOR = 4
CODE_FINGERPRINT_PATHS = (
    COMMON_ROOT / "nidsbed" / "adaptation.py",
    COMMON_ROOT / "nidsbed" / "datasets.py",
    COMMON_ROOT / "nidsbed" / "model_selection.py",
    COMMON_ROOT / "nidsbed" / "algorithms" / "erm.py",
    PROJECT_ROOT / "CODE" / "experiment_low_label" / "run_finetune_from_checkpoint.py",
    PROJECT_ROOT / "CODE" / "experiment_stage3" / "train.py",
)


class NoFiniteValidationMetrics(RuntimeError):
    def __init__(self, mode: str, history: list[dict]):
        super().__init__(f"no finite validation AUROC/AUPRC for mode={mode}")
        self.history = history


class SingleClassSupportError(RuntimeError):
    """Raised after persisting a passive support that cannot support selection."""


def resolve_support_policy(support_policy: str) -> tuple[str, bool]:
    if support_policy == "balanced":
        return FEW_SHOT_POLICY_GROUP_NESTED, True
    if support_policy == "random_group":
        return FEW_SHOT_POLICY_GROUP_RANDOM_NESTED, False
    raise ValueError(f"unsupported support_policy={support_policy!r}")


def support_policy_output_root(output_root: Path, support_policy: str) -> Path:
    resolve_support_policy(support_policy)
    if support_policy == "balanced":
        return output_root
    return output_root / f"support_policy={support_policy}"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_lr_grid(values: list[float]) -> list[float]:
    grid = sorted(set(float(value) for value in values))
    if not grid or not np.isfinite(grid).all() or any(value <= 0.0 for value in grid):
        raise ValueError("lr_grid must contain finite positive values")
    return grid


def validation_candidate_is_better(
    auroc: float,
    auprc: float,
    tie_break_value: float,
    incumbent: tuple[float, float, float] | None,
    atol: float = SELECTION_ATOL,
) -> bool:
    """Compare support-validation metrics with a deterministic stable tie-break."""
    if not np.isfinite([auroc, auprc, tie_break_value]).all():
        return False
    if incumbent is None:
        return True
    best_auroc, best_auprc, best_tie_break = incumbent
    if auroc > best_auroc + atol:
        return True
    if abs(auroc - best_auroc) > atol:
        return False
    if auprc > best_auprc + atol:
        return True
    if abs(auprc - best_auprc) > atol:
        return False
    return tie_break_value < best_tie_break


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in CODE_FINGERPRINT_PATHS:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_schema_fingerprint(checkpoint: dict) -> str:
    schema = {
        "algorithm": checkpoint.get("algorithm"),
        "in_dim": int(checkpoint["in_dim"]),
        "n_classes": int(checkpoint["n_classes"]),
        "hparams": checkpoint["hparams"],
    }
    encoded = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run_dir(output_root: Path, target: str, n: int, mode: str,
            checkpoint_seed: int, support_seed: int,
            adaptation_seed: int, support_policy: str = "balanced") -> Path:
    return (
        support_policy_output_root(output_root, support_policy)
        / target
        / f"n{n}"
        / mode
        / (
            f"ckpt{checkpoint_seed}_support{support_seed}_"
            f"adapt{adaptation_seed}"
        )
    )


def load_checkpoint(target: str, seed: int, device: str,
                    template: str | None = None) -> tuple[dict, Path]:
    if template is None:
        path = Path(str(DEFAULT_CHECKPOINT_TEMPLATE).format(target=target, seed=seed))
    else:
        path = Path(template.format(target=target, seed=seed)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("algorithm") != "erm":
        raise ValueError(f"revision runner requires ERM checkpoint, got {checkpoint.get('algorithm')}")
    if checkpoint.get("target_domain") != target:
        raise ValueError(
            f"checkpoint target mismatch: expected {target}, got "
            f"{checkpoint.get('target_domain')}"
        )
    return checkpoint, path


def build_algorithm(checkpoint: dict, mode: str, class_weights: dict,
                    device: str, model_seed: int):
    seed_everything(model_seed)
    if mode == "target_only":
        algo_cls = get_algorithm_class("erm")
        algorithm = algo_cls(
            in_dim=int(checkpoint["in_dim"]),
            n_classes=int(checkpoint["n_classes"]),
            hparams=dict(checkpoint["hparams"]),
            class_weights_per_domain=class_weights,
            device=device,
        )
        return algorithm
    return _build_algorithm_from_checkpoint_payload(
        checkpoint,
        device,
        class_weights,
    )


def invariant_checks(mode: str, configured: dict, final: dict) -> dict:
    checks = {}
    if mode in {"true_linear_probe", "head_tune_warmstart"}:
        checks["backbone_parameters_frozen"] = (
            configured["backbone_parameters_sha256"]
            == final["backbone_parameters_sha256"]
        )
        checks["backbone_buffers_frozen"] = (
            configured["backbone_buffers_sha256"]
            == final["backbone_buffers_sha256"]
        )
    if mode == "bn_recal":
        checks["all_parameters_frozen"] = (
            configured["parameters_sha256"] == final["parameters_sha256"]
        )
    if mode == "bn_only":
        checks["classifier_frozen"] = (
            configured["classifier_parameters_sha256"]
            == final["classifier_parameters_sha256"]
        )
    if checks and not all(checks.values()):
        raise AssertionError(f"adaptation state invariant failed for {mode}: {checks}")
    return checks


def train_adaptation(algorithm, mode: str, train_dataset,
                     val_X: np.ndarray, val_y: np.ndarray,
                     device: str, optimization_seed: int, lr: float,
                     max_epochs: int, patience: int,
                     finetune_dataset=None) -> tuple[dict, list[dict]]:
    seed_everything(optimization_seed)
    configured_hashes = model_state_hashes(algorithm)
    history = []

    if mode == "bn_recal":
        meta = recalibrate_batch_norm(
            algorithm,
            train_dataset.X.cpu().numpy(),
            device=device,
        )
        metrics = evaluate_full(algorithm, val_X, val_y, device=device)
        return {
            "best_epoch": 0,
            "best_val_auroc": metrics["auroc"],
            "best_val_auprc": metrics["auprc"],
            "best_val_metrics": metrics,
            "bn_recalibration": meta,
            "configured_hashes": configured_hashes,
        }, history

    n_fit = len(train_dataset)
    batch_size = max(8, min(256, n_fit // 4))
    iterator = GPUResidentBatchIterator(
        [train_dataset],
        batch_size=batch_size,
        domain_balanced=False,
        device=device,
        seed=optimization_seed,
    )
    if len(iterator) < 1:
        raise ValueError(
            f"fit support is too small for batch_size={batch_size}: n_fit={n_fit}"
        )
    probe_updates_per_epoch = int(len(iterator))
    finetune_batch_size = int(batch_size)
    finetune_updates_per_epoch = int(len(iterator))
    best_score = None
    best_epoch = -1
    best_stage = None
    best_state = None
    stale = 0

    def run_epoch(epoch: int, stage: str, optimizer,
                  l2sp_reference=None, l2sp_coefficient: float = 0.0) -> bool:
        nonlocal best_score, best_epoch, best_stage, best_state, stale
        losses = []
        for batch in iterator:
            losses.append(adaptation_train_step(
                algorithm,
                batch,
                mode,
                optimizer,
                stage=stage,
                l2sp_reference=l2sp_reference,
                l2sp_coefficient=l2sp_coefficient,
            ))
        val_metrics = evaluate_full(algorithm, val_X, val_y, device=device)
        val_auroc = float(val_metrics["auroc"])
        val_auprc = float(val_metrics["auprc"])
        history.append({
            "epoch": epoch,
            "stage": stage,
            "train_loss": float(np.mean(losses)),
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
            "n_updates": len(losses),
        })
        candidate_score = (val_auroc, val_auprc, float(epoch))
        if validation_candidate_is_better(*candidate_score, best_score):
            best_score = candidate_score
            best_epoch = epoch
            best_stage = stage
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in algorithm.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        return stale >= patience

    if mode in LPFT_MODES:
        source_backbone = capture_backbone_reference(algorithm)
        probe_epochs = min(LPFT_PROBE_EPOCHS, max_epochs)
        optimizer = build_adaptation_optimizer(
            algorithm, lr=lr, weight_decay=1e-5, mode=mode, stage="probe"
        )
        algorithm.optimizer = optimizer
        for epoch in range(1, probe_epochs + 1):
            run_epoch(epoch, "probe", optimizer)

        if probe_epochs < max_epochs:
            configure_lpft_finetune(algorithm, mode)
            if finetune_dataset is not None:
                n_finetune = len(finetune_dataset)
                finetune_batch_size = max(8, min(256, n_finetune // 4))
                iterator = GPUResidentBatchIterator(
                    [finetune_dataset],
                    batch_size=finetune_batch_size,
                    domain_balanced=False,
                    device=device,
                    seed=optimization_seed,
                )
                if len(iterator) < 1:
                    raise ValueError(
                        "target-guided replay dataset is too small for "
                        f"batch_size={finetune_batch_size}: n={n_finetune}"
                    )
                finetune_updates_per_epoch = int(len(iterator))
            optimizer = build_adaptation_optimizer(
                algorithm,
                lr=lr,
                weight_decay=1e-5,
                mode=mode,
                stage="finetune",
            )
            algorithm.optimizer = optimizer
            stale = 0
            coefficient = (
                LPFT_L2SP_COEFFICIENT
                if mode == "lpft_frozen_bn_l2sp"
                else 0.0
            )
            for epoch in range(probe_epochs + 1, max_epochs + 1):
                should_stop = run_epoch(
                    epoch,
                    "finetune",
                    optimizer,
                    l2sp_reference=source_backbone,
                    l2sp_coefficient=coefficient,
                )
                if should_stop:
                    break
    else:
        optimizer = build_adaptation_optimizer(
            algorithm, lr=lr, weight_decay=1e-5, mode=mode
        )
        algorithm.optimizer = optimizer
        for epoch in range(1, max_epochs + 1):
            if run_epoch(epoch, "finetune", optimizer):
                break

    if best_state is None:
        raise NoFiniteValidationMetrics(mode, history)
    algorithm.load_state_dict(best_state)
    algorithm.eval()
    best_val_metrics = evaluate_full(algorithm, val_X, val_y, device=device)
    return {
        "best_epoch": int(best_epoch),
        "best_stage": best_stage,
        "best_val_auroc": float(best_score[0]),
        "best_val_auprc": float(best_val_metrics["auprc"]),
        "epoch_selection_tie_break": (
            f"metrics_equal_within_{SELECTION_ATOL:g}_prefer_earlier_epoch"
        ),
        "best_val_metrics": best_val_metrics,
        "batch_size": int(batch_size),
        "updates_per_epoch": int(finetune_updates_per_epoch),
        "probe_updates_per_epoch": probe_updates_per_epoch,
        "finetune_batch_size": finetune_batch_size,
        "finetune_updates_per_epoch": finetune_updates_per_epoch,
        "total_updates": int(sum(item["n_updates"] for item in history)),
        "lpft_probe_epochs": (
            min(LPFT_PROBE_EPOCHS, max_epochs) if mode in LPFT_MODES else 0
        ),
        "lpft_backbone_l2sp_coefficient": (
            LPFT_L2SP_COEFFICIENT
            if mode == "lpft_frozen_bn_l2sp"
            else 0.0
        ),
        "configured_hashes": configured_hashes,
    }, history


def operational_metrics(algorithm, loader: LODODataLoader,
                        val_X: np.ndarray, val_y: np.ndarray,
                        eval_scores: np.ndarray, eval_y: np.ndarray,
                        device: str) -> tuple[dict, dict]:
    val_logits = predict_logits_batched(algorithm, val_X, device=device)
    val_scores = scores_from_logits(val_logits, temperature=1.0)

    source_per_domain = loader.get_source_val_per_domain()
    source_scores = {}
    source_labels = {}
    for domain, (source_X, source_y) in source_per_domain.items():
        logits = predict_logits_batched(algorithm, source_X, device=device)
        source_scores[domain] = scores_from_logits(logits, temperature=1.0)
        source_labels[domain] = source_y

    pooled_scores = np.concatenate([source_scores[d] for d in loader.source_domains])
    pooled_labels = np.concatenate([source_labels[d] for d in loader.source_domains])
    output = {"source_pooled": {}, "source_worst_domain": {}, "support_val": {}}

    for budget in FPR_BUDGETS:
        key = f"fpr_{budget:g}"
        pooled_selection = select_threshold_at_max_fpr(
            pooled_labels, pooled_scores, budget
        )
        output["source_pooled"][key] = {
            "selection": pooled_selection,
            "target_fixed_test": metrics_at_fixed_threshold(
                eval_y, eval_scores, pooled_selection["threshold"]
            ),
        }

        per_domain = {
            domain: select_threshold_at_max_fpr(
                source_labels[domain], source_scores[domain], budget
            )
            for domain in loader.source_domains
        }
        worst_threshold = max(item["threshold"] for item in per_domain.values())
        output["source_worst_domain"][key] = {
            "per_domain_selection": per_domain,
            "threshold": float(worst_threshold),
            "target_fixed_test": metrics_at_fixed_threshold(
                eval_y, eval_scores, worst_threshold
            ),
        }

        support_selection = select_threshold_at_max_fpr(val_y, val_scores, budget)
        output["support_val"][key] = {
            "selection": support_selection,
            "target_fixed_test": metrics_at_fixed_threshold(
                eval_y, eval_scores, support_selection["threshold"]
            ),
        }

    return output, {
        "support_val_logits": val_logits.astype(np.float32),
        "support_val_scores": val_scores.astype(np.float32),
        "support_val_labels": val_y.astype(np.int32),
    }


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def build_target_guided_replay_dataset(
    loader: LODODataLoader,
    target_fit_dataset,
    device: str,
    neighbors_per_target: int = REPLAY_NEIGHBORS_PER_TARGET_PER_SOURCE,
    target_repeat_factor: int = REPLAY_TARGET_REPEAT_FACTOR,
) -> tuple[LODODataset, dict]:
    """Build target-fit-guided replay without reading target validation/test."""
    target_x = target_fit_dataset.X.cpu().numpy()
    target_y = target_fit_dataset.y.cpu().numpy()
    source_by_domain = loader.get_source_train(per_domain=True)
    replay_x = []
    replay_y = []
    domain_audit = {}
    for domain in loader.source_domains:
        source_x, source_y = source_by_domain[domain]
        split = load_domain_split(loader.data_dir, domain)
        source_group_ids = split["row_group_id"][split["train_idx"]]
        selected, audit = select_label_matched_source_replay(
            source_x,
            source_y,
            source_group_ids,
            target_x,
            target_y,
            neighbors_per_target=neighbors_per_target,
            device=device,
        )
        replay_x.append(np.asarray(source_x[selected], dtype=np.float32))
        replay_y.append(np.asarray(source_y[selected], dtype=np.int64))
        domain_audit[domain] = {
            **audit,
            "selected_train_positions_sha256": _array_sha256(selected),
            "selected_group_ids_sha256": _array_sha256(
                np.asarray(source_group_ids[selected])
            ),
        }

    source_x = np.concatenate(replay_x, axis=0)
    source_y = np.concatenate(replay_y, axis=0)
    mixed_x = np.concatenate(
        [np.repeat(target_x, int(target_repeat_factor), axis=0), source_x], axis=0
    )
    mixed_y = np.concatenate(
        [np.repeat(target_y, int(target_repeat_factor)), source_y], axis=0
    )
    audit = {
        "protocol": "target_fit_same_label_nearest_source_group_replay_v1",
        "neighbors_per_target_per_source_domain": int(neighbors_per_target),
        "target_repeat_factor": int(target_repeat_factor),
        "target_fit_rows": int(len(target_x)),
        "source_replay_rows": int(len(source_x)),
        "mixed_rows": int(len(mixed_x)),
        "source_domains": domain_audit,
        "target_fit_x_sha256": _array_sha256(target_x),
        "source_replay_x_sha256": _array_sha256(source_x),
        "source_replay_y_sha256": _array_sha256(source_y),
    }
    return LODODataset(mixed_x, mixed_y, domain_label=0), audit


def run_one(loader: LODODataLoader, checkpoint: dict, checkpoint_path: Path,
            checkpoint_hash: str, target: str, n: int, mode: str,
            checkpoint_seed: int, support_seed: int, model_seed: int,
            optimization_seed: int,
            device: str, lr_grid: list[float], max_epochs: int, patience: int,
            code_fingerprint: str, output_dir: Path,
            support_policy: str = "balanced") -> dict:
    started = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    dataset_support_policy, support_balanced = resolve_support_policy(support_policy)
    loader.configure_few_shot_split(
        n=n,
        seed=support_seed,
        balanced=support_balanced,
        policy=dataset_support_policy,
    )
    train_dataset = loader.get_few_shot_dataset(domain_label=0, subset="train")
    val_X, val_y = loader.get_few_shot_validation()
    eval_X, eval_y = loader.get_cross_domain_eval_excluding_few_shot()
    eval_X_raw, _ = loader.get_cross_domain_eval_unscaled_excluding_few_shot()
    train_y = train_dataset.y.cpu().numpy()
    support_audit = loader.get_few_shot_audit()
    partition_counts = {
        "fit": {"0": int((train_y == 0).sum()), "1": int((train_y == 1).sum())},
        "validation": {"0": int((val_y == 0).sum()), "1": int((val_y == 1).sum())},
    }
    single_class_partitions = [
        name for name, counts in partition_counts.items()
        if counts["0"] == 0 or counts["1"] == 0
    ]
    if single_class_partitions:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npz(
            output_dir / "support_split.npz",
            selected_indices=loader._few_shot_idx.astype(np.int64),
            selected_group_ids=np.asarray(loader._few_shot_group_ids),
            class_ranks=loader._few_shot_class_ranks.astype(np.int64),
            train_mask=loader._few_shot_train_mask.astype(bool),
            val_mask=loader._few_shot_val_mask.astype(bool),
            policy=np.asarray(loader.target_few_shot_policy),
            support_policy=np.asarray(support_policy),
            support_seed=np.asarray(support_seed, dtype=np.int64),
        )
        failure = {
            "status": "failed_single_class_support",
            "failure_type": "single_class_support_partition",
            "failure_message": (
                "passively sampled support has a single-class fit or validation "
                "partition; the persisted support is not resampled"
            ),
            "single_class_partitions": single_class_partitions,
            "support_partition_class_counts": partition_counts,
            "protocol_version": ADAPTATION_PROTOCOL_VERSION,
            "target": target,
            "n_labeled": int(n),
            "mode": mode,
            "checkpoint_seed": int(checkpoint_seed),
            "support_seed": int(support_seed),
            "model_seed": int(model_seed),
            "optimization_seed": int(optimization_seed),
            "adaptation_seed": int(optimization_seed),
            "support_policy": support_policy,
            "support_sampling_policy": dataset_support_policy,
            "support_balanced": support_balanced,
            "support_audit": support_audit,
            "code_fingerprint_sha256": code_fingerprint,
        }
        atomic_write_json(output_dir / "result.json", failure)
        raise SingleClassSupportError(
            f"single-class support partitions={single_class_partitions} "
            f"counts={partition_counts}; support persisted without resampling"
        )
    replay_dataset = None
    replay_audit = None
    if mode == "target_guided_replay_lpft":
        replay_dataset, replay_audit = build_target_guided_replay_dataset(
            loader,
            train_dataset,
            device=device,
        )
        class_weights = {
            0: get_class_weights(
                replay_dataset.y.cpu().numpy(), device=device
            )
        }
    else:
        class_weights = {0: get_class_weights(train_y, device=device)}

    candidate_lrs = [None] if mode in DIAGNOSTIC_ONLY_MODES else list(lr_grid)
    if not candidate_lrs:
        raise ValueError("lr_grid must contain at least one learning rate")
    search_summaries = []
    search_histories = {}
    selected = None
    selected_score = None
    for candidate_lr in candidate_lrs:
        algorithm = build_algorithm(
            checkpoint,
            mode,
            class_weights,
            device,
            model_seed,
        )
        initial_hashes = model_state_hashes(algorithm)
        mode_manifest = configure_adaptation_mode(algorithm, mode, model_seed)
        configured_hashes = model_state_hashes(algorithm)
        try:
            train_info, history = train_adaptation(
                algorithm,
                mode,
                train_dataset,
                val_X,
                val_y,
                device,
                optimization_seed,
                float(candidate_lr or 0.0),
                max_epochs,
                patience,
                finetune_dataset=replay_dataset,
            )
        except NoFiniteValidationMetrics as error:
            lr_key = "none" if candidate_lr is None else f"{candidate_lr:.12g}"
            search_histories[lr_key] = error.history
            search_summaries.append({
                "lr": candidate_lr,
                "status": "unselectable_nonfinite_validation",
                "best_epoch": None,
                "best_val_auroc": None,
                "best_val_auprc": None,
                "elapsed_epochs": len(error.history),
            })
            del algorithm
            continue
        final_hashes = model_state_hashes(algorithm)
        checks = invariant_checks(mode, configured_hashes, final_hashes)
        lr_key = "none" if candidate_lr is None else f"{candidate_lr:.12g}"
        search_histories[lr_key] = history
        summary = {
            "lr": candidate_lr,
            "best_epoch": train_info["best_epoch"],
            "best_val_auroc": train_info["best_val_auroc"],
            "best_val_auprc": train_info["best_val_auprc"],
            "elapsed_epochs": len(history),
        }
        search_summaries.append(summary)
        score = (
            float(train_info["best_val_auroc"]),
            float(train_info["best_val_auprc"]),
            float(candidate_lr or 0.0),
        )
        if validation_candidate_is_better(*score, selected_score):
            selected_score = score
            selected = {
                "algorithm": algorithm,
                "lr": candidate_lr,
                "manifest": mode_manifest,
                "train_info": train_info,
                "history": history,
                "initial_hashes": initial_hashes,
                "configured_hashes": configured_hashes,
                "final_hashes": final_hashes,
                "checks": checks,
            }
        else:
            del algorithm

    if selected is None:
        raise RuntimeError(f"no selectable LR candidate for mode={mode}")
    algorithm = selected["algorithm"]
    selected_lr = selected["lr"]
    mode_manifest = selected["manifest"]
    train_info = selected["train_info"]
    history = selected["history"]
    initial_hashes = selected["initial_hashes"]
    configured_hashes = selected["configured_hashes"]
    final_hashes = selected["final_hashes"]
    checks = selected["checks"]

    eval_logits = predict_logits_batched(algorithm, eval_X, device=device)
    eval_scores = scores_from_logits(eval_logits, temperature=1.0)
    eval_metrics = evaluate_full(
        algorithm,
        eval_X,
        eval_y,
        device=device,
        unique_weights=unique_row_weights(eval_X_raw),
    )
    deployment, support_artifacts = operational_metrics(
        algorithm,
        loader,
        val_X,
        val_y,
        eval_scores,
        eval_y,
        device,
    )

    result = {
        "status": "success",
        "protocol_version": ADAPTATION_PROTOCOL_VERSION,
        "target": target,
        "n_labeled": int(n),
        "mode": mode,
        "support_policy": support_policy,
        "support_sampling_policy": dataset_support_policy,
        "support_balanced": support_balanced,
        "checkpoint_seed": int(checkpoint_seed),
        "support_seed": int(support_seed),
        "model_seed": int(model_seed),
        "optimization_seed": int(optimization_seed),
        "adaptation_seed": int(optimization_seed),
        "seed_design": "model_and_optimization_seed_crossed_with_support_v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_schema_sha256": checkpoint_schema_fingerprint(checkpoint),
        "code_fingerprint_sha256": code_fingerprint,
        "checkpoint_weight_dependent": mode != "target_only",
        "checkpoint_dependent": mode != "target_only",
        "checkpoint_schema_dependent": True,
        "source_information_scope": (
            "shared_features_and_source_selected_model_config_without_source_weights"
            if mode == "target_only"
            else (
                "shared_features_source_selected_model_config_source_weights_and_"
                "target_fit_selected_source_train_replay_rows"
                if mode == "target_guided_replay_lpft"
                else "shared_features_source_selected_model_config_and_source_weights"
            )
        ),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "support_audit": support_audit,
        "target_guided_replay": replay_audit,
        "adaptation_manifest": mode_manifest,
        "training": train_info,
        "state_hashes": {
            "initial": initial_hashes,
            "configured": configured_hashes,
            "final": final_hashes,
            "invariant_checks": checks,
        },
        "target_fixed_test": eval_metrics,
        "deployment": deployment,
        "selected_lr": selected_lr,
        "lr_grid": [float(value) for value in lr_grid],
        "lr_selection_metric": "support_val_auroc_then_auprc",
        "lr_selection_tie_break": (
            f"metrics_equal_within_{SELECTION_ATOL:g}_prefer_smaller_lr"
        ),
        "lr_search": search_summaries,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "elapsed_sec": round(time.time() - started, 3),
        "peak_cuda_memory_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3)
            if device.startswith("cuda") else None
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(
        output_dir / "support_split.npz",
        selected_indices=loader._few_shot_idx.astype(np.int64),
        selected_group_ids=np.asarray(loader._few_shot_group_ids),
        class_ranks=loader._few_shot_class_ranks.astype(np.int64),
        train_mask=loader._few_shot_train_mask.astype(bool),
        val_mask=loader._few_shot_val_mask.astype(bool),
        policy=np.asarray(loader.target_few_shot_policy),
        support_policy=np.asarray(support_policy),
        support_seed=np.asarray(support_seed, dtype=np.int64),
    )
    atomic_save_npz(
        output_dir / "predictions.npz",
        target_eval_logits=eval_logits.astype(np.float32),
        target_eval_scores=eval_scores.astype(np.float32),
        target_eval_labels=eval_y.astype(np.int32),
        **support_artifacts,
    )
    atomic_torch_save(
        output_dir / "adapted_checkpoint.pt",
        {
            "protocol_version": ADAPTATION_PROTOCOL_VERSION,
            "target": target,
            "mode": mode,
            "support_policy": support_policy,
            "support_sampling_policy": dataset_support_policy,
            "support_balanced": support_balanced,
            "checkpoint_seed": checkpoint_seed,
            "support_seed": support_seed,
            "model_seed": model_seed,
            "optimization_seed": optimization_seed,
            "adaptation_seed": optimization_seed,
            "seed_design": "model_and_optimization_seed_crossed_with_support_v1",
            "n_labeled": n,
            "state_dict": {
                key: value.detach().cpu() for key, value in algorithm.state_dict().items()
            },
            "in_dim": checkpoint["in_dim"],
            "n_classes": checkpoint["n_classes"],
            "hparams": checkpoint["hparams"],
            "support_audit": support_audit,
            "target_guided_replay": replay_audit,
            "adaptation_manifest": mode_manifest,
            "selected_lr": selected_lr,
            "code_fingerprint_sha256": code_fingerprint,
            "lr_search": search_summaries,
        },
    )
    atomic_write_json(output_dir / "history.json", {"history": history})
    atomic_write_json(
        output_dir / "lr_search.json",
        {"candidates": search_summaries, "histories": search_histories},
    )
    atomic_write_json(output_dir / "result.json", result)
    return result


def result_is_complete(path: Path, expected: dict | None = None) -> bool:
    result_path = path / "result.json"
    required = [
        result_path,
        path / "support_split.npz",
        path / "predictions.npz",
        path / "adapted_checkpoint.pt",
        path / "history.json",
        path / "lr_search.json",
    ]
    if any(not item.exists() or item.stat().st_size == 0 for item in required):
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    complete = (
        payload.get("status") == "success"
        and payload.get("protocol_version") == ADAPTATION_PROTOCOL_VERSION
        and payload.get("support_audit", {}).get("support_eval_group_overlap") == 0
    )
    if not complete or expected is None:
        return complete
    return all(payload.get(key) == value for key, value in expected.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--checkpoint_seed", type=int, required=True)
    parser.add_argument("--support_seed", type=int, required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--adaptation_seed", type=int, required=True)
    parser.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument(
        "--support_policy", choices=SUPPORT_POLICIES, default="balanced"
    )
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "DATA" / "nfv2_processed"))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checkpoint_template", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--lr_grid",
        default=(
            "0.00001,0.00003,0.0001,0.0003,0.001,0.003,"
            "0.01,0.03,0.1,0.3,1.0,3.0"
        ),
        help="Equal-budget LR candidates selected only by held-out support validation.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Compatibility shortcut that overrides --lr_grid with one LR.",
    )
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ns = parse_int_list(args.ns)
    lr_grid = normalize_lr_grid(
        [args.lr] if args.lr is not None else parse_float_list(args.lr_grid)
    )
    modes = parse_str_list(args.modes)
    invalid_modes = sorted(set(modes) - ALL_ADAPTATION_MODES)
    if invalid_modes:
        raise ValueError(f"unsupported modes: {invalid_modes}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root).expanduser().resolve()

    planned = [
        run_dir(
            output_root,
            args.target,
            n,
            mode,
            args.checkpoint_seed,
            args.support_seed,
            args.adaptation_seed,
            args.support_policy,
        )
        for n in ns
        for mode in modes
    ]
    if args.dry_run:
        for path in planned:
            print(path)
        return

    checkpoint, checkpoint_path = load_checkpoint(
        args.target,
        args.checkpoint_seed,
        device,
        template=args.checkpoint_template,
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    code_fingerprint = experiment_code_fingerprint()
    dataset_support_policy, support_balanced = resolve_support_policy(
        args.support_policy
    )
    loader = LODODataLoader(
        data_dir=args.data_dir,
        target_domain=args.target,
        use_invalid_indicators=True,
        target_few_shot_n=0,
        target_few_shot_balanced=support_balanced,
        target_few_shot_policy=dataset_support_policy,
    )
    loader.prepare()

    print(
        f"[{args.target}] device={device} checkpoint_seed={args.checkpoint_seed} "
        f"support_seed={args.support_seed} model_seed={args.model_seed} "
        f"optimization_seed={args.adaptation_seed} "
        f"support_policy={args.support_policy}"
    )
    failures = []
    for n in ns:
        for mode in modes:
            path = run_dir(
                output_root,
                args.target,
                n,
                mode,
                args.checkpoint_seed,
                args.support_seed,
                args.adaptation_seed,
                args.support_policy,
            )
            expected = {
                "target": args.target,
                "n_labeled": n,
                "mode": mode,
                "support_policy": args.support_policy,
                "support_sampling_policy": resolve_support_policy(args.support_policy)[0],
                "support_balanced": resolve_support_policy(args.support_policy)[1],
                "checkpoint_seed": args.checkpoint_seed,
                "support_seed": args.support_seed,
                "model_seed": args.model_seed,
                "optimization_seed": args.adaptation_seed,
                "adaptation_seed": args.adaptation_seed,
                "seed_design": "model_and_optimization_seed_crossed_with_support_v1",
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_schema_sha256": checkpoint_schema_fingerprint(checkpoint),
                "code_fingerprint_sha256": code_fingerprint,
                "lr_grid": [float(value) for value in lr_grid],
                "lr_selection_tie_break": (
                    f"metrics_equal_within_{SELECTION_ATOL:g}_prefer_smaller_lr"
                ),
                "max_epochs": args.max_epochs,
                "patience": args.patience,
            }
            if not args.force and result_is_complete(path, expected):
                print(f"  skip n={n} mode={mode}: complete")
                continue
            print(f"  run  n={n} mode={mode}")
            try:
                result = run_one(
                    loader,
                    checkpoint,
                    checkpoint_path,
                    checkpoint_hash,
                    args.target,
                    n,
                    mode,
                    args.checkpoint_seed,
                    args.support_seed,
                    args.model_seed,
                    args.adaptation_seed,
                    device,
                    lr_grid,
                    args.max_epochs,
                    args.patience,
                    code_fingerprint,
                    path,
                    args.support_policy,
                )
            except SingleClassSupportError as error:
                failures.append(str(path))
                print(f"       FAILED: {error}")
                continue
            metrics = result["target_fixed_test"]
            print(
                f"       AUROC={metrics['auroc']:.4f} "
                f"AUPRC={metrics['auprc']:.4f} "
                f"F1={metrics['attack_f1']:.4f} "
                f"lr={result['selected_lr']} "
                f"time={result['elapsed_sec']:.1f}s"
            )
    if failures:
        raise SystemExit(
            f"{len(failures)} run(s) failed because passive support contained a "
            "single-class fit or validation partition; see their result.json files"
        )


if __name__ == "__main__":
    main()
