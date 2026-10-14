"""Explicit, auditable adaptation modes for target-support experiments."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


ADAPTATION_PROTOCOL_VERSION = "group_fixed_test_adaptation_v5"
TRAINING_MODES = {
    "full_ft",
    "true_linear_probe",
    "head_tune_warmstart",
    "bn_only",
    "target_only",
    "lpft_frozen_bn",
    "lpft_train_bn",
    "lpft_frozen_bn_l2sp",
    "warm_head_ft_frozen_bn",
    "warm_head_ft_train_bn",
    "target_guided_replay_lpft",
}
DIAGNOSTIC_ONLY_MODES = {"bn_recal"}
ALL_ADAPTATION_MODES = TRAINING_MODES | DIAGNOSTIC_ONLY_MODES
LPFT_MODES = {
    "lpft_frozen_bn",
    "lpft_train_bn",
    "lpft_frozen_bn_l2sp",
    "warm_head_ft_frozen_bn",
    "warm_head_ft_train_bn",
    "target_guided_replay_lpft",
}
LPFT_BACKBONE_LR_RATIO = 0.1
LPFT_L2SP_COEFFICIENT = 1e-3


def batch_norm_modules(module: nn.Module) -> list[nn.Module]:
    return [
        child
        for child in module.modules()
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]


def _reset_classifier(classifier: nn.Module, seed: int) -> None:
    devices = []
    if next(classifier.parameters()).is_cuda:
        devices = [next(classifier.parameters()).device.index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        for module in classifier.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def configure_adaptation_mode(algorithm: nn.Module, mode: str,
                              adaptation_seed: int) -> dict:
    """Configure trainable parameters and initialization for one named mode."""
    if mode not in ALL_ADAPTATION_MODES:
        raise ValueError(f"unsupported adaptation mode={mode!r}")

    for parameter in algorithm.parameters():
        parameter.requires_grad = False

    head_init = "checkpoint"
    bn_policy = "checkpoint_frozen"
    if mode in {"full_ft", "target_only"}:
        for parameter in algorithm.parameters():
            parameter.requires_grad = True
        bn_policy = "checkpoint_initialized_train_updates"
        if mode == "target_only":
            head_init = "random_model"
    elif mode in LPFT_MODES:
        if mode.startswith("lpft_") or mode == "target_guided_replay_lpft":
            _reset_classifier(algorithm.classifier, adaptation_seed)
            head_init = "deterministic_reset"
        else:
            head_init = "checkpoint"
        for parameter in algorithm.classifier.parameters():
            parameter.requires_grad = True
        bn_policy = (
            "checkpoint_frozen_during_probe_then_train_updates"
            if mode in {"lpft_train_bn", "warm_head_ft_train_bn"}
            else "checkpoint_frozen_during_probe_and_finetune"
        )
    elif mode == "true_linear_probe":
        _reset_classifier(algorithm.classifier, adaptation_seed)
        for parameter in algorithm.classifier.parameters():
            parameter.requires_grad = True
        head_init = "deterministic_reset"
    elif mode == "head_tune_warmstart":
        for parameter in algorithm.classifier.parameters():
            parameter.requires_grad = True
    elif mode == "bn_only":
        for bn in batch_norm_modules(algorithm.backbone):
            if bn.affine:
                bn.weight.requires_grad = True
                bn.bias.requires_grad = True
        bn_policy = "checkpoint_initialized_affine_and_buffers_update"
    elif mode == "bn_recal":
        bn_policy = "reset_buffers_fit_support_single_pass"

    set_adaptation_train_mode(algorithm, mode, stage="probe")
    trainable_names = [
        name for name, parameter in algorithm.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "protocol_version": ADAPTATION_PROTOCOL_VERSION,
        "mode": mode,
        "head_init": head_init,
        "bn_policy": bn_policy,
        "trainable_parameter_names": trainable_names,
        "n_trainable_parameters": int(sum(
            parameter.numel()
            for parameter in algorithm.parameters()
            if parameter.requires_grad
        )),
        "lpft": mode in LPFT_MODES,
        "lpft_backbone_lr_ratio": (
            LPFT_BACKBONE_LR_RATIO if mode in LPFT_MODES else None
        ),
        "lpft_l2sp_coefficient": (
            LPFT_L2SP_COEFFICIENT if mode == "lpft_frozen_bn_l2sp" else None
        ),
    }


def set_adaptation_train_mode(algorithm: nn.Module, mode: str,
                              stage: str = "finetune") -> None:
    """Apply module train/eval states without silently changing the protocol."""
    if mode in {"full_ft", "target_only"}:
        algorithm.train()
    elif mode in LPFT_MODES:
        if stage == "probe":
            algorithm.eval()
            algorithm.classifier.train()
        elif stage == "finetune":
            algorithm.train()
            if mode not in {"lpft_train_bn", "warm_head_ft_train_bn"}:
                for bn in batch_norm_modules(algorithm.backbone):
                    bn.eval()
        else:
            raise ValueError(f"unsupported LP-FT stage={stage!r}")
    elif mode in {"true_linear_probe", "head_tune_warmstart"}:
        algorithm.eval()
        algorithm.classifier.train()
    elif mode == "bn_only":
        algorithm.eval()
        for bn in batch_norm_modules(algorithm.backbone):
            bn.train()
    elif mode == "bn_recal":
        algorithm.eval()
    else:
        raise ValueError(f"unsupported adaptation mode={mode!r}")


def configure_lpft_finetune(algorithm: nn.Module, mode: str) -> None:
    """Unfreeze LP-FT after its head-only warm-up without resetting weights."""
    if mode not in LPFT_MODES:
        raise ValueError(f"not an LP-FT mode: {mode!r}")
    for parameter in algorithm.parameters():
        parameter.requires_grad = True
    set_adaptation_train_mode(algorithm, mode, stage="finetune")


def build_adaptation_optimizer(algorithm: nn.Module, lr: float,
                               weight_decay: float = 1e-5,
                               mode: str | None = None,
                               stage: str = "finetune"):
    if mode in LPFT_MODES and stage == "finetune":
        groups = [
            {
                "params": [p for p in algorithm.classifier.parameters() if p.requires_grad],
                "lr": float(lr),
            },
            {
                "params": [p for p in algorithm.backbone.parameters() if p.requires_grad],
                "lr": float(lr) * LPFT_BACKBONE_LR_RATIO,
            },
        ]
        return torch.optim.Adam(groups, weight_decay=float(weight_decay))
    params = [parameter for parameter in algorithm.parameters() if parameter.requires_grad]
    if not params:
        return None
    return torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))


def capture_backbone_reference(algorithm: nn.Module) -> dict[str, torch.Tensor]:
    """Capture source backbone weights for an LP-FT L2-SP penalty."""
    return {
        name: parameter.detach().clone()
        for name, parameter in algorithm.backbone.named_parameters()
    }


def l2sp_backbone_penalty(algorithm: nn.Module,
                          reference: dict[str, torch.Tensor] | None,
                          coefficient: float) -> torch.Tensor:
    if reference is None or coefficient <= 0.0:
        return torch.zeros((), device=next(algorithm.parameters()).device)
    penalty = torch.zeros((), device=next(algorithm.parameters()).device)
    for name, parameter in algorithm.backbone.named_parameters():
        penalty = penalty + (parameter - reference[name]).pow(2).sum()
    return float(coefficient) * penalty


def select_label_matched_source_replay(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_group_ids: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    neighbors_per_target: int,
    device: str,
    chunk_size: int = 16384,
) -> tuple[np.ndarray, dict]:
    """Select nearest same-label source representatives for target-guided replay.

    Source rows are first reduced to one row per exact-feature group. The target
    test is not an input: all queries must be fit-support representatives.
    """
    if neighbors_per_target < 1:
        raise ValueError("neighbors_per_target must be positive")
    source_x = np.asarray(source_x, dtype=np.float32)
    source_y = np.asarray(source_y, dtype=np.int64)
    source_group_ids = np.asarray(source_group_ids)
    target_x = np.asarray(target_x, dtype=np.float32)
    target_y = np.asarray(target_y, dtype=np.int64)
    if len(source_x) != len(source_y) or len(source_x) != len(source_group_ids):
        raise ValueError("source replay arrays have inconsistent lengths")
    if len(target_x) != len(target_y):
        raise ValueError("target replay arrays have inconsistent lengths")

    _, representative_idx = np.unique(source_group_ids, return_index=True)
    representative_idx = np.sort(representative_idx.astype(np.int64, copy=False))
    chosen: list[np.ndarray] = []
    by_class = {}
    for label in sorted(np.unique(target_y).tolist()):
        target_mask = target_y == label
        candidate_idx = representative_idx[source_y[representative_idx] == label]
        if not len(candidate_idx):
            raise ValueError(f"no source replay candidates for target label={label}")
        query = torch.from_numpy(target_x[target_mask]).to(device)
        best_dist = torch.full(
            (len(query), neighbors_per_target), float("inf"), device=device
        )
        best_idx = torch.full(
            (len(query), neighbors_per_target), -1, dtype=torch.int64, device=device
        )
        for start in range(0, len(candidate_idx), chunk_size):
            chunk_idx = candidate_idx[start : start + chunk_size]
            candidate = torch.from_numpy(source_x[chunk_idx]).to(device)
            distances = torch.cdist(query, candidate)
            n_take = min(neighbors_per_target, distances.shape[1])
            chunk_dist, chunk_pos = torch.topk(
                distances, k=n_take, largest=False, dim=1
            )
            chunk_source_idx = torch.from_numpy(chunk_idx).to(device)[chunk_pos]
            combined_dist = torch.cat([best_dist, chunk_dist], dim=1)
            combined_idx = torch.cat([best_idx, chunk_source_idx], dim=1)
            best_dist, keep = torch.topk(
                combined_dist, k=neighbors_per_target, largest=False, dim=1
            )
            best_idx = torch.gather(combined_idx, 1, keep)
        selected = best_idx.detach().cpu().numpy().reshape(-1)
        selected = selected[selected >= 0]
        chosen.append(selected)
        by_class[str(label)] = {
            "n_target_fit": int(target_mask.sum()),
            "n_source_candidate_groups": int(len(candidate_idx)),
            "n_selected_before_union": int(len(selected)),
        }

    selected_idx = np.unique(np.concatenate(chosen)).astype(np.int64, copy=False)
    selected_groups = source_group_ids[selected_idx]
    if len(np.unique(selected_groups)) != len(selected_groups):
        raise AssertionError("source replay contains repeated exact groups")
    if not np.isin(source_y[selected_idx], np.unique(target_y)).all():
        raise AssertionError("source replay contains unavailable target labels")
    return selected_idx, {
        "policy": "same_label_nearest_source_group_replay_v1",
        "neighbors_per_target_per_source_domain": int(neighbors_per_target),
        "source_groups_before_selection": int(len(representative_idx)),
        "source_groups_selected": int(len(selected_idx)),
        "per_class": by_class,
    }


def adaptation_train_step(algorithm: nn.Module, batch: tuple,
                          mode: str, optimizer, stage: str = "finetune",
                          l2sp_reference: dict[str, torch.Tensor] | None = None,
                          l2sp_coefficient: float = 0.0) -> float:
    """Run one ERM adaptation step while preserving mode-specific module state."""
    if optimizer is None:
        raise ValueError(f"mode {mode!r} has no trainable parameters")
    set_adaptation_train_mode(algorithm, mode, stage=stage)
    x, y, domain_labels = batch[:3]

    if mode in {"true_linear_probe", "head_tune_warmstart"} or (
        mode in LPFT_MODES and stage == "probe"
    ):
        with torch.no_grad():
            features = algorithm.backbone(x)
    else:
        features = algorithm.backbone(x)
    logits = algorithm.classifier(features)
    loss = algorithm.compute_per_domain_cls_loss(logits, y, domain_labels)
    if mode == "lpft_frozen_bn_l2sp" and stage == "finetune":
        loss = loss + l2sp_backbone_penalty(
            algorithm, l2sp_reference, l2sp_coefficient
        )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def recalibrate_batch_norm(algorithm: nn.Module, x_fit: np.ndarray,
                           device: str) -> dict:
    """Reset and estimate BN buffers from fit-support while Dropout stays disabled."""
    if len(x_fit) < 2:
        raise ValueError("BN recalibration requires at least two fit-support rows")
    bn_modules = batch_norm_modules(algorithm.backbone)
    if not bn_modules:
        return {"n_bn_modules": 0, "n_fit": int(len(x_fit))}

    algorithm.eval()
    saved_momenta = []
    for bn in bn_modules:
        saved_momenta.append(bn.momentum)
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()

    x_tensor = torch.from_numpy(np.asarray(x_fit, dtype=np.float32)).to(device)
    algorithm.backbone(x_tensor)

    for bn, momentum in zip(bn_modules, saved_momenta):
        bn.momentum = momentum
        bn.eval()
    algorithm.eval()
    return {
        "n_bn_modules": int(len(bn_modules)),
        "n_fit": int(len(x_fit)),
        "policy": "reset_buffers_fit_support_single_pass_dropout_eval_v1",
    }


def _hash_tensors(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_state_hashes(algorithm: nn.Module) -> dict[str, str]:
    """Hash parameters and buffers separately for freeze-invariant auditing."""
    return {
        "parameters_sha256": _hash_tensors(algorithm.named_parameters()),
        "buffers_sha256": _hash_tensors(algorithm.named_buffers()),
        "backbone_parameters_sha256": _hash_tensors(
            algorithm.backbone.named_parameters(prefix="backbone")
        ),
        "backbone_buffers_sha256": _hash_tensors(
            algorithm.backbone.named_buffers(prefix="backbone")
        ),
        "classifier_parameters_sha256": _hash_tensors(
            algorithm.classifier.named_parameters(prefix="classifier")
        ),
    }
