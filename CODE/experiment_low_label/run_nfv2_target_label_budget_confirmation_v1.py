"""Low-budget extension for the frozen NFv2 target-label budget study.

This runner is deliberately source-scheduled: learning rate and epoch horizon are
read from frozen source ERM checkpoints before the target loader is prepared.  It
does not select on target-support validation and never reads external datasets.
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))

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

from nidsbed.calibration import binary_ece, scores_from_logits
from nidsbed.adaptation import model_state_hashes
from nidsbed.datasets import (
    FEW_SHOT_POLICY_GROUP_NESTED,
    GPUResidentBatchIterator,
    LODODataLoader,
    unique_row_weights,
)
from nidsbed.lib.misc import get_class_weights, seed_everything
from nidsbed.model_selection import (
    evaluate_score_arrays,
    metrics_at_fixed_threshold,
    predict_logits_batched,
    select_threshold_at_max_fpr,
)
from nidsbed.diagnostic_helpers import select_weighted_threshold_at_max_fpr, weighted_fixed_threshold
from CODE.experiment_low_label.run_fixed_schedule_sensitivity import build_algorithm, train_fixed_schedule


PROTOCOL = "nfv2_target_label_budget_confirmation_v1"
DOMAINS = ("unsw", "cic", "bot", "ton")
MODES = ("full_ft", "target_only")
CHECKPOINT_SEEDS = (2026, 2027, 2028)
SUPPORT_SEEDS = (11, 23, 37, 53, 71)
LOW_BUDGETS = (10, 20)
FPR_BUDGETS = (0.01, 0.001, 0.0001)
DEFAULT_OUTPUT = PROJECT_ROOT / "RESULTS" / "low_label" / "target_label_budget_confirmation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def weighted_metrics(y: np.ndarray, scores: np.ndarray, threshold: float, weights: np.ndarray) -> dict:
    return weighted_fixed_threshold(y, scores, threshold, weights)


def source_schedule(stage3_root: Path, target: str, seed: int) -> tuple[dict, dict, Path]:
    fold = stage3_root / f"erm_{target}" / "final"
    checkpoint_path = fold / f"seed{seed}_artifacts" / "checkpoint.pt"
    hp_path = fold / "best_hp.json"
    if not checkpoint_path.is_file() or not hp_path.is_file():
        raise FileNotFoundError(f"missing frozen source artifacts for {target}/seed{seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hp = json.loads(hp_path.read_text(encoding="utf-8"))
    if checkpoint.get("algorithm") != "erm" or checkpoint.get("target_domain") != target:
        raise ValueError(f"source checkpoint metadata mismatch: {checkpoint_path}")
    if checkpoint.get("hparams") != hp.get("hparams"):
        raise ValueError(f"source checkpoint/hparams mismatch: {checkpoint_path}")
    schedule = {
        "fixed_lr": float(checkpoint["hparams"]["lr"]),
        "fixed_epochs": int(checkpoint["best_epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_hp_sha256": sha256_file(hp_path),
    }
    return checkpoint, schedule, checkpoint_path


def run_coordinate(
    *,
    target: str,
    n: int,
    mode: str,
    checkpoint_seed: int,
    support_seed: int,
    loader: LODODataLoader,
    checkpoint: dict,
    schedule: dict,
    output_root: Path,
    device: str,
    force: bool,
) -> dict:
    run_dir = output_root / target / f"n{n}" / mode / f"ckpt{checkpoint_seed}_support{support_seed}"
    result_path = run_dir / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    loader.configure_few_shot_split(
        n=n, seed=support_seed, balanced=True, policy=FEW_SHOT_POLICY_GROUP_NESTED
    )
    support = loader.get_few_shot_dataset(domain_label=0, subset="train")
    if support is None or len(support) == 0:
        raise ValueError(f"empty target support for {target}, N={n}")
    class_weights = {0: get_class_weights(support.y.cpu().numpy(), device=device)}
    algorithm = build_algorithm(checkpoint, mode, class_weights, device, checkpoint_seed)
    initial_state_hashes = model_state_hashes(algorithm)
    training, history = train_fixed_schedule(
        algorithm,
        mode=mode,
        train_dataset=support,
        device=device,
        optimization_seed=checkpoint_seed,
        lr=schedule["fixed_lr"],
        fixed_epochs=schedule["fixed_epochs"],
    )
    eval_X, eval_y = loader.get_cross_domain_eval_excluding_few_shot()
    eval_X_raw, _ = loader.get_cross_domain_eval_unscaled_excluding_few_shot()
    eval_logits = predict_logits_batched(algorithm, eval_X, batch_size=16384, device=device)
    eval_scores = scores_from_logits(eval_logits, temperature=1.0)
    eval_weights = unique_row_weights(eval_X_raw)
    target_metrics = evaluate_score_arrays(
        eval_y, eval_scores, preds=np.argmax(eval_logits, axis=1), unique_weights=eval_weights
    )

    source_val = loader.get_source_val_per_domain()
    source_raw = loader.get_source_val_unscaled(per_domain=True)
    source_labels = []
    source_weights = []
    for domain in loader.source_domains:
        sx, sy = source_val[domain]
        source_labels.append(np.asarray(sy, dtype=np.int64))
        source_weights.append(unique_row_weights(source_raw[domain][0]))
    source_logits = []
    for domain in loader.source_domains:
        sx, _ = source_val[domain]
        source_logits.append(predict_logits_batched(algorithm, sx, batch_size=16384, device=device))
    source_logits = np.concatenate(source_logits)
    pooled_scores = scores_from_logits(source_logits, temperature=1.0)
    pooled_labels = np.concatenate(source_labels)
    pooled_weights = np.concatenate(source_weights)
    deployment = {}
    for budget in FPR_BUDGETS:
        tag = f"fpr_{budget:g}"
        row_sel = select_threshold_at_max_fpr(pooled_labels, pooled_scores, budget)
        group_sel = select_weighted_threshold_at_max_fpr(
            pooled_labels, pooled_scores, pooled_weights, budget
        )
        deployment[tag] = {
            "row": {
                "selection": row_sel,
                "target": metrics_at_fixed_threshold(eval_y, eval_scores, row_sel["threshold"]),
            },
            "group": {
                "selection": group_sel,
                "target": weighted_metrics(eval_y, eval_scores, group_sel["threshold"], eval_weights),
            },
        }

    # Calibration is fitted on pooled source validation only; target labels are post-hoc.
    from nidsbed.calibration import fit_temperature_from_logits
    temperature, calibration_meta = fit_temperature_from_logits(source_logits, pooled_labels)
    target_cal = scores_from_logits(eval_logits, temperature=temperature)
    calibration = {
        "temperature": float(temperature),
        "source_brier": float(np.mean((scores_from_logits(source_logits, temperature=temperature) - pooled_labels) ** 2)),
        "target_brier": float(np.mean((target_cal - eval_y) ** 2)),
        "source_ece": float(binary_ece(scores_from_logits(source_logits, temperature=temperature), pooled_labels)),
        "target_ece": float(binary_ece(target_cal, eval_y)),
        "fit_metadata": calibration_meta,
    }
    latency_rows = eval_X[: min(10000, len(eval_X))]
    for _ in range(2):
        predict_logits_batched(algorithm, latency_rows, batch_size=16384, device=device)
    t0 = time.perf_counter()
    predict_logits_batched(algorithm, latency_rows, batch_size=16384, device=device)
    elapsed_infer = time.perf_counter() - t0
    adapted_state_hashes = model_state_hashes(algorithm)
    support_audit = loader.get_few_shot_audit()
    if support_audit.get("support_eval_group_overlap") != 0:
        raise ValueError("support/test duplicate-group overlap")
    result = {
        "status": "success",
        "protocol": PROTOCOL,
        "target": target,
        "n_labeled": int(n),
        "mode": mode,
        "checkpoint_seed": int(checkpoint_seed),
        "support_seed": int(support_seed),
        "model_seed": int(checkpoint_seed),
        "optimization_seed": int(checkpoint_seed),
        "source_schedule": {**schedule, "target_support_validation_used": False, "target_test_used": False},
        "support_audit": support_audit,
        "target_fixed_test": target_metrics,
        "deployment": deployment,
        "calibration": calibration,
        "training": training,
        "n_training_history": len(history),
        "initial_state_hashes": initial_state_hashes,
        "adapted_state_hashes": adapted_state_hashes,
        "inference": {
            "n_rows_benchmark": int(len(latency_rows)),
            "latency_ms_per_1000_rows": float(elapsed_infer * 1000.0 / max(len(latency_rows), 1) * 1000.0),
            "throughput_rows_per_sec": float(len(latency_rows) / max(elapsed_infer, 1e-12)),
        },
        "target_labels_used_for_training_or_selection": True,
        "target_labels_used_for_model_or_threshold_selection": False,
        "external_data_used": False,
        "elapsed_sec": float(time.time() - started),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
    }
    atomic_npz(
        run_dir / "predictions.npz",
        target_eval_logits=eval_logits.astype(np.float32),
        target_eval_scores=eval_scores.astype(np.float32),
        target_eval_labels=eval_y.astype(np.int32),
        target_eval_unique_weights=eval_weights.astype(np.float64),
    )
    atomic_npz(
        run_dir / "support_split.npz",
        selected_indices=loader._few_shot_idx.astype(np.int64),
        selected_group_ids=np.asarray(loader._few_shot_group_ids),
        class_ranks=loader._few_shot_class_ranks.astype(np.int64),
        train_mask=loader._few_shot_train_mask.astype(bool),
        val_mask=loader._few_shot_val_mask.astype(bool),
        policy=np.asarray(loader.target_few_shot_policy),
        support_seed=np.asarray(support_seed, dtype=np.int64),
    )
    atomic_json(run_dir / "history.json", {"history": history})
    result["artifact_sha256"] = {
        name: sha256_file(run_dir / name)
        for name in ("predictions.npz", "support_split.npz", "history.json")
    }
    atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "DATA" / "nfv2_processed")
    parser.add_argument("--stage3-root", type=Path, default=PROJECT_ROOT / "RESULTS" / "stage3_lodo" / "stage3")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", default=",".join(DOMAINS))
    parser.add_argument("--budgets", default=",".join(map(str, LOW_BUDGETS)))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--checkpoint-seeds", default=",".join(map(str, CHECKPOINT_SEEDS)))
    parser.add_argument("--support-seeds", default=",".join(map(str, SUPPORT_SEEDS)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = tuple(x.strip() for x in args.targets.split(",") if x.strip())
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    checkpoint_seeds = tuple(int(x.strip()) for x in args.checkpoint_seeds.split(",") if x.strip())
    support_seeds = tuple(int(x.strip()) for x in args.support_seeds.split(",") if x.strip())
    if any(n < 2 or n % 2 for n in budgets):
        raise ValueError("budgets must be positive even group counts")
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol": PROTOCOL,
        "targets": list(targets),
        "budgets": list(budgets),
        "modes": list(modes),
        "checkpoint_seeds": list(checkpoint_seeds),
        "support_seeds": list(support_seeds),
        "source_schedule_only": True,
        "target_support_validation_used_for_selection": False,
        "external_data_used": False,
    }
    atomic_json(args.output_root / "plan.json", plan)
    total = len(targets) * len(budgets) * len(modes) * len(checkpoint_seeds) * len(support_seeds)
    completed = 0
    for target in targets:
        for checkpoint_seed in checkpoint_seeds:
            checkpoint, schedule, _ = source_schedule(args.stage3_root, target, checkpoint_seed)
            loader = LODODataLoader(data_dir=str(args.data_dir), target_domain=target, target_few_shot_n=0)
            loader.prepare()
            for n in budgets:
                for mode in modes:
                    for support_seed in support_seeds:
                        run_coordinate(
                            target=target, n=n, mode=mode,
                            checkpoint_seed=checkpoint_seed, support_seed=support_seed,
                            loader=loader, checkpoint=checkpoint, schedule=schedule,
                            output_root=args.output_root, device=args.device, force=args.force,
                        )
                        completed += 1
                        print(json.dumps({"completed": completed, "total": total, "target": target, "n": n, "mode": mode, "checkpoint_seed": checkpoint_seed, "support_seed": support_seed}), flush=True)
    atomic_json(args.output_root / "run_status.json", {**plan, "status": "success", "n_coordinates": completed})


if __name__ == "__main__":
    main()
