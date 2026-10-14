"""Fixed-schedule helpers restored from the original experiment implementation.

Only the two helpers used by the N=10/20 runner are included. Their function
bodies are unchanged; this is not the old standalone sensitivity launcher.
Source: paper/scripts/run_fixed_schedule_sensitivity.py
Source SHA256: 02908283467dbd0979b421facfe014e9dfdaae7d49496bc686c0f889cda3a5ac
"""
from __future__ import annotations

import numpy as np
from nidsbed.adaptation import adaptation_train_step, build_adaptation_optimizer
from nidsbed.datasets import GPUResidentBatchIterator
from nidsbed.lib.misc import seed_everything
from CODE.experiment_stage3.train import _build_algorithm_from_checkpoint_payload, get_algorithm_class

DEFAULT_MODES = ("full_ft", "target_only")
ADAPTATION_WEIGHT_DECAY = 1e-5
SCHEDULE_POLICY = "transport_checkpoint_source_val_lr_and_best_epoch_v1"
FINAL_STATE_POLICY = "state_after_last_planned_update_no_best_state_reload_v1"


class FixedScheduleError(RuntimeError):
    """Invalid fixed-schedule input or failed optimization."""


def build_algorithm(
    checkpoint: dict,
    mode: str,
    class_weights: dict,
    device: str,
    model_seed: int,
):
    seed_everything(model_seed)
    if mode == "target_only":
        algorithm_class = get_algorithm_class("erm")
        return algorithm_class(
            in_dim=int(checkpoint["in_dim"]),
            n_classes=int(checkpoint["n_classes"]),
            hparams=dict(checkpoint["hparams"]),
            class_weights_per_domain=class_weights,
            device=device,
        )
    if mode != "full_ft":
        raise FixedScheduleError(f"unsupported fixed-schedule mode: {mode}")
    return _build_algorithm_from_checkpoint_payload(checkpoint, device, class_weights)


def train_fixed_schedule(
    algorithm,
    *,
    mode: str,
    train_dataset,
    device: str,
    optimization_seed: int,
    lr: float,
    fixed_epochs: int,
) -> tuple[dict, list[dict]]:
    """Train to the final planned update without validation or rollback."""
    if mode not in DEFAULT_MODES:
        raise FixedScheduleError(f"unsupported fixed-schedule mode: {mode}")
    seed_everything(optimization_seed)
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
        raise FixedScheduleError(f"empty fixed-schedule iterator for n_fit={n_fit}")
    optimizer = build_adaptation_optimizer(
        algorithm,
        lr=float(lr),
        weight_decay=ADAPTATION_WEIGHT_DECAY,
    )
    algorithm.optimizer = optimizer
    history = []
    total_updates = 0
    for epoch in range(1, int(fixed_epochs) + 1):
        losses = []
        for batch in iterator:
            loss = adaptation_train_step(algorithm, batch, mode, optimizer)
            if not np.isfinite(loss):
                raise FixedScheduleError(
                    f"non-finite loss at epoch={epoch}, update={len(losses) + 1}"
                )
            losses.append(float(loss))
        total_updates += len(losses)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "n_updates": len(losses),
            "cumulative_updates": total_updates,
        })
    algorithm.eval()
    return {
        "policy": SCHEDULE_POLICY,
        "fixed_lr": float(lr),
        "fixed_epochs": int(fixed_epochs),
        "batch_size": int(batch_size),
        "updates_per_epoch": int(len(iterator)),
        "observed_epochs": len(history),
        "observed_total_updates": int(total_updates),
        "final_state_policy": FINAL_STATE_POLICY,
        "early_stopping_enabled": False,
        "target_support_validation_evaluations": 0,
        "best_state_reload": False,
    }, history
