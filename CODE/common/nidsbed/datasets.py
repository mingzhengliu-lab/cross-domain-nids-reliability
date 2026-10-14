"""LODO data loading, duplicate-aware splits, and batch sampling.

- Leave-one-domain-out: 3 source domains train/val/test-internal + 1 target.
- Per-domain 80/10/10 splits with exact-feature-row grouping.
- StandardScaler and invalid indicators are fit on source train only.
- DomainBalancedBatchSampler keeps each batch balanced across source domains.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler

from .data_utils import (exact_row_group_ids, unique_row_weights, load_domain_split,
                         _invalid_mask, fit_source_train_preprocessor, apply_source_train_preprocessor)

from .attack_family_ontology import (
    ONTOLOGY_VERSION,
    SHARED4_ONTOLOGY_VERSION,
    coarse_attack_family,
    shared4_attack_family,
)


# ============================================================
# ============================================================

ALL_DOMAINS = ['unsw', 'cic', 'bot', 'ton']

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_INTERNAL_FRAC = 0.10  # train + val + test_internal = 1.00

LODO_SPLIT_SEED = 2026
SPLIT_POLICY = "duplicate_aware_exact_feature_group_v4"
SPLIT_ALGO_VERSION = "hybrid_duplicate_group_assignment_v2"
SPLIT_ASSIGNMENT_MODE = "best_of_majority_class_size_greedy_and_joint_row_attack_greedy"
SPLIT_OBJECTIVE_VERSION = "size_and_label_ratio_objective_v2"
FEATURE_PREPROCESS_POLICY = "source_train_quantile_cap_scale_invalid_indicators_v1"
SPLIT_DIRNAME = "splits"
SPLIT_CANDIDATE_SEEDS = 8
SPLIT_LABEL_RATIO_WARN_THRESHOLD = 0.03
INVALID_SENTINEL_ABS = 1e9
CAP_LOWER_Q = 0.001
CAP_UPPER_Q = 0.999
SCALE_CLIP = 10.0

FEW_SHOT_POLICY_LEGACY = "row_stratified_v1"
FEW_SHOT_POLICY_GROUP_NESTED = "duplicate_group_balanced_nested_v1"
FEW_SHOT_POLICY_GROUP_RANDOM_NESTED = "duplicate_group_random_nested_v1"
GROUP_AWARE_FEW_SHOT_POLICIES = frozenset({
    FEW_SHOT_POLICY_GROUP_NESTED,
    FEW_SHOT_POLICY_GROUP_RANDOM_NESTED,
})
SUPPORTED_FEW_SHOT_POLICIES = frozenset({
    FEW_SHOT_POLICY_LEGACY,
    *GROUP_AWARE_FEW_SHOT_POLICIES,
})
DEFAULT_FEW_SHOT_POLICY = FEW_SHOT_POLICY_GROUP_NESTED
FEW_SHOT_VAL_EVERY = 5
SUPPORTED_ATTACK_FAMILY_POLICIES = frozenset({"exact", "coarse13_v1", "shared4_v1"})


def _validate_few_shot_policy_balance(policy: str, balanced: bool) -> None:
    if policy not in SUPPORTED_FEW_SHOT_POLICIES:
        raise ValueError(f"unsupported target_few_shot_policy={policy!r}")
    if policy == FEW_SHOT_POLICY_GROUP_NESTED and not balanced:
        raise ValueError(
            f"policy {policy!r} requires balanced=True; use "
            f"{FEW_SHOT_POLICY_GROUP_RANDOM_NESTED!r} for passive uniform-group sampling"
        )
    if policy == FEW_SHOT_POLICY_GROUP_RANDOM_NESTED and balanced:
        raise ValueError(f"policy {policy!r} requires balanced=False")


def _atomic_write_json_if_changed(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        if path.read_text(encoding="utf-8") == serialized:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(serialized, encoding="utf-8")
    os.replace(tmp, path)


# ============================================================
# ============================================================





def build_group_nested_few_shot_split(
    y: np.ndarray,
    group_ids: np.ndarray,
    n: int,
    seed: int,
    balanced: bool = True,
    val_every: int = FEW_SHOT_VAL_EVERY,
    candidate_mask: np.ndarray | None = None,
    evaluation_mask: np.ndarray | None = None,
    policy: str | None = None,
) -> dict[str, np.ndarray | int | bool | str]:
    """Build a leakage-free, nested target support split.

    ``n`` counts labelled duplicate groups, not raw rows. One representative
    row is labelled from each selected group, while every row in those groups
    is removed from target evaluation. Sampling permutations depend only on
    ``seed`` so increasing ``n`` takes a strict prefix of the same stream. For
    class-balanced sampling, every ``val_every``-th group in each class is
    assigned to validation. For passive uniform-group sampling, acquisition
    and validation assignment use one global group permutation and do not use
    labels. Both designs keep the fit and validation subsets nested as ``n``
    increases.
    """
    y = np.asarray(y, dtype=np.int64)
    group_ids = np.asarray(group_ids)
    n = int(n)
    seed = int(seed)
    val_every = int(val_every)
    balanced = bool(balanced)
    if policy is None:
        policy = (
            FEW_SHOT_POLICY_GROUP_NESTED
            if balanced
            else FEW_SHOT_POLICY_GROUP_RANDOM_NESTED
        )
    _validate_few_shot_policy_balance(policy, balanced)
    if policy not in GROUP_AWARE_FEW_SHOT_POLICIES:
        raise ValueError(
            "build_group_nested_few_shot_split requires a group-aware policy, "
            f"got {policy!r}"
        )

    if y.ndim != 1 or group_ids.ndim != 1 or len(y) != len(group_ids):
        raise ValueError("y and group_ids must be aligned one-dimensional arrays")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if val_every < 2:
        raise ValueError(f"val_every must be >=2, got {val_every}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("few-shot sampling requires binary labels 0/1")

    if candidate_mask is None:
        candidate_mask = np.ones(len(y), dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != y.shape:
            raise ValueError("candidate_mask must have the same shape as y")
    candidate_idx = np.flatnonzero(candidate_mask)
    if candidate_idx.size == 0:
        raise ValueError("candidate_mask selects no support candidates")

    candidate_groups = group_ids[candidate_idx]
    candidate_y = y[candidate_idx]
    unique_groups, first_local_idx, inverse = np.unique(
        candidate_groups,
        return_index=True,
        return_inverse=True,
    )
    first_idx = candidate_idx[first_local_idx]
    n_groups = len(unique_groups)
    group_min = np.full(n_groups, 1, dtype=np.int8)
    group_max = np.full(n_groups, 0, dtype=np.int8)
    np.minimum.at(group_min, inverse, candidate_y.astype(np.int8, copy=False))
    np.maximum.at(group_max, inverse, candidate_y.astype(np.int8, copy=False))
    mixed = np.flatnonzero(group_min != group_max)
    if balanced and mixed.size:
        raise ValueError(
            "duplicate groups with mixed binary labels are not supported: "
            f"{mixed.size} mixed groups"
        )
    group_labels = y[first_idx]

    selected_dense_parts = []
    selected_rank_parts = []
    if balanced:
        wanted = {0: n - (n // 2), 1: n // 2}
        for cls in (0, 1):
            candidates = np.flatnonzero(group_labels == cls)
            if wanted[cls] > len(candidates):
                raise ValueError(
                    f"requested {wanted[cls]} class-{cls} groups, only "
                    f"{len(candidates)} are available"
                )
            rng = np.random.default_rng(np.random.SeedSequence([seed, 104729 + cls]))
            ordered = candidates[rng.permutation(len(candidates))]
            selected_dense_parts.append(ordered[:wanted[cls]])
            selected_rank_parts.append(np.arange(wanted[cls], dtype=np.int64))
    else:
        if n > n_groups:
            raise ValueError(f"requested {n} groups, only {n_groups} are available")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 999983]))
        ordered = rng.permutation(n_groups)[:n]
        selected_dense_parts.append(ordered)
        selected_rank_parts.append(np.arange(n, dtype=np.int64))

    selected_dense = np.concatenate(selected_dense_parts).astype(np.int64, copy=False)
    class_ranks = np.concatenate(selected_rank_parts).astype(np.int64, copy=False)
    selected_indices = first_idx[selected_dense].astype(np.int64, copy=False)
    selected_group_ids = unique_groups[selected_dense]
    selected_labels = group_labels[selected_dense].astype(np.int64, copy=False)
    val_mask = ((class_ranks + 1) % val_every) == 0
    train_mask = ~val_mask

    # Preserve the passive sampling stream so its selected arrays are strict
    # prefixes across budgets. The established balanced policy retains its
    # stable row-index artifact order.
    order = (
        np.argsort(selected_indices, kind="stable")
        if balanced
        else np.arange(len(selected_indices), dtype=np.int64)
    )
    selected_indices = selected_indices[order]
    selected_group_ids = selected_group_ids[order]
    selected_labels = selected_labels[order]
    class_ranks = class_ranks[order]
    train_mask = train_mask[order]
    val_mask = val_mask[order]

    if evaluation_mask is None:
        eval_mask = ~np.isin(group_ids, selected_group_ids)
        evaluation_policy = "full_target_excluding_selected_groups"
    else:
        eval_mask = np.asarray(evaluation_mask, dtype=bool).copy()
        if eval_mask.shape != y.shape:
            raise ValueError("evaluation_mask must have the same shape as y")
        evaluation_policy = "fixed_external_mask"
    if np.intersect1d(group_ids[eval_mask], selected_group_ids).size:
        raise ValueError("support duplicate groups overlap the requested evaluation mask")

    selected_group_row_mask = np.isin(group_ids, selected_group_ids)

    return {
        "policy": policy,
        "selected_indices": selected_indices,
        "selected_group_ids": selected_group_ids,
        "selected_labels": selected_labels,
        "class_ranks": class_ranks,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "eval_mask": eval_mask,
        "n_selected_groups": int(len(selected_group_ids)),
        "n_selected_group_rows": int(selected_group_row_mask.sum()),
        "n_selected_sibling_rows": int(selected_group_row_mask.sum() - len(selected_indices)),
        "n_evaluation_rows": int(eval_mask.sum()),
        "evaluation_policy": evaluation_policy,
        "n_candidate_rows": int(candidate_mask.sum()),
        "balanced": balanced,
        "sampling_design": (
            "class_balanced_duplicate_groups"
            if balanced
            else "uniform_candidate_duplicate_groups"
        ),
        "selection_uses_labels": balanced,
        "validation_rank_scope": "per_class" if balanced else "global_stream",
        "validation_assignment": (
            f"per_class_rank_every_{val_every}"
            if balanced
            else f"global_acquisition_rank_every_{val_every}"
        ),
        "seed": seed,
        "val_every": val_every,
    }


def _split_indices_from_groups(X: np.ndarray, y: np.ndarray, seed: int,
                               n_candidates: int = SPLIT_CANDIDATE_SEEDS) -> tuple[dict, dict]:
    group_ids, group_counts = exact_row_group_ids(X)
    n_groups = int(len(group_counts))
    group_index = np.arange(n_groups)
    group_attack = np.bincount(group_ids, weights=(y == 1).astype(np.float64), minlength=n_groups)
    group_y = (group_attack >= (group_counts / 2.0)).astype(np.int64)
    mixed_groups = int(((group_attack > 0) & (group_attack < group_counts)).sum())

    def _build(candidate_seed: int, candidate_rank: int) -> tuple[dict, dict]:
        split_by_group = np.full(n_groups, -1, dtype=np.int8)
        rng = np.random.RandomState(candidate_seed)
        target_frac = np.array([TRAIN_FRAC, VAL_FRAC, TEST_INTERNAL_FRAC], dtype=np.float64)

        if candidate_rank == 0:
            # Conservative baseline: split majority-benign and majority-attack
            # duplicate groups separately. This preserves 80/10/10 sizes well
            # on ordinary domains and remains a candidate for the objective.
            for cls in [0, 1]:
                cls_groups = group_index[group_y == cls]
                if len(cls_groups) == 0:
                    continue
                shuffled_cls = cls_groups[rng.permutation(len(cls_groups))]
                ordered_cls = shuffled_cls[np.argsort(-group_counts[shuffled_cls])].tolist()
                target_rows_cls = float(group_counts[cls_groups].sum()) * target_frac
                current_rows_cls = np.zeros(3, dtype=np.float64)
                for g in ordered_cls:
                    deficits = target_rows_cls - current_rows_cls
                    norm_deficits = deficits / np.maximum(target_rows_cls, 1.0)
                    split = int(np.argmax(norm_deficits))
                    split_by_group[g] = split
                    current_rows_cls[split] += float(group_counts[g])
        else:
            shuffled = group_index[rng.permutation(n_groups)]
            priority = group_counts[shuffled].astype(np.float64) * rng.uniform(
                0.60, 1.40, size=len(shuffled)
            )
            ordered = shuffled[np.argsort(-priority)].tolist()

            target_rows = float(len(y)) * target_frac
            target_attacks = float((y == 1).sum()) * target_frac
            target_benign = target_rows - target_attacks
            current_rows = np.zeros(3, dtype=np.float64)
            current_attacks = np.zeros(3, dtype=np.float64)

            for g in ordered:
                rows_g = float(group_counts[g])
                attacks_g = float(group_attack[g])
                best_split = 0
                best_loss = float("inf")
                for split in range(3):
                    loss = 0.0
                    for i in range(3):
                        rows_i = current_rows[i] + (rows_g if i == split else 0.0)
                        attacks_i = current_attacks[i] + (attacks_g if i == split else 0.0)
                        benign_i = rows_i - attacks_i
                        row_err = ((rows_i - target_rows[i]) / max(target_rows[i], 1.0)) ** 2
                        attack_err = ((attacks_i - target_attacks[i]) / max(target_attacks[i], 1.0)) ** 2
                        benign_err = ((benign_i - target_benign[i]) / max(target_benign[i], 1.0)) ** 2
                        loss += row_err + 3.0 * attack_err + 3.0 * benign_err
                    if loss < best_loss:
                        best_split = split
                        best_loss = loss
                split_by_group[g] = best_split
                current_rows[best_split] += rows_g
                current_attacks[best_split] += attacks_g

        if np.any(split_by_group < 0):
            split_by_group[split_by_group < 0] = 0

        train_idx = np.flatnonzero(split_by_group[group_ids] == 0).astype(np.int64)
        val_idx = np.flatnonzero(split_by_group[group_ids] == 1).astype(np.int64)
        test_idx = np.flatnonzero(split_by_group[group_ids] == 2).astype(np.int64)

        splits = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "row_group_id": group_ids.astype(np.int32, copy=False),
        }

        def _ratio(idx):
            return float(y[idx].mean()) if len(idx) else float("nan")

        full_ratio = float(y.mean())
        split_defs = [
            ("train", train_idx, TRAIN_FRAC),
            ("val", val_idx, VAL_FRAC),
            ("test_internal", test_idx, TEST_INTERNAL_FRAC),
        ]
        split_info = {}
        ratio_devs = []
        size_devs = []
        for name, idx, target_size_frac in split_defs:
            ratio = _ratio(idx)
            ratio_dev = abs(ratio - full_ratio) if np.isfinite(ratio) else float("nan")
            size_frac = float(len(idx) / max(1, len(y)))
            size_dev = abs(size_frac - target_size_frac)
            split_info[name] = {
                "n": int(len(idx)),
                "target_size_fraction": float(target_size_frac),
                "size_fraction": size_frac,
                "size_fraction_deviation": size_dev,
                "attack_ratio": ratio,
                "attack_ratio_deviation": ratio_dev,
            }
            ratio_devs.append(ratio_dev if np.isfinite(ratio_dev) else 1.0)
            size_devs.append(size_dev)

        group_sets = {
            "train": set(np.unique(group_ids[train_idx]).tolist()),
            "val": set(np.unique(group_ids[val_idx]).tolist()),
            "test": set(np.unique(group_ids[test_idx]).tolist()),
        }
        max_ratio_dev = float(max(ratio_devs)) if ratio_devs else 0.0
        max_size_dev = float(max(size_devs)) if size_devs else 0.0
        objective = float(10.0 * max_ratio_dev + 2.0 * sum(ratio_devs) +
                          max_size_dev + 0.1 * sum(size_devs))
        audit = {
            "policy": SPLIT_POLICY,
            "split_policy": SPLIT_POLICY,
            "split_algo_version": SPLIT_ALGO_VERSION,
            "split_assignment_mode_family": SPLIT_ASSIGNMENT_MODE,
            "split_objective_version": SPLIT_OBJECTIVE_VERSION,
            "seed": int(seed),
            "chosen_seed": int(candidate_seed),
            "candidate_rank": int(candidate_rank),
            "split_assignment_mode": (
                "majority_class_size_greedy"
                if candidate_rank == 0 else "joint_row_attack_greedy"
            ),
            "candidate_seeds_tried": int(n_candidates),
            "objective_value": objective,
            "split_objective": objective,
            "max_attack_ratio_deviation": max_ratio_dev,
            "max_size_fraction_deviation": max_size_dev,
            "label_ratio_warning_threshold": SPLIT_LABEL_RATIO_WARN_THRESHOLD,
            "label_ratio_warning": bool(max_ratio_dev > SPLIT_LABEL_RATIO_WARN_THRESHOLD),
            "n_rows": int(len(y)),
            "n_unique_feature_rows": n_groups,
            "duplicate_ratio": float(1.0 - n_groups / max(1, len(y))),
            "mixed_label_duplicate_groups": mixed_groups,
            "full_attack_ratio": full_ratio,
            "splits": split_info,
            "exact_feature_group_overlap": {
                "train_val": len(group_sets["train"] & group_sets["val"]),
                "train_test": len(group_sets["train"] & group_sets["test"]),
                "val_test": len(group_sets["val"] & group_sets["test"]),
            },
        }
        return splits, audit

    best_splits = None
    best_audit = None
    best_objective = float("inf")
    for offset in range(max(1, int(n_candidates))):
        candidate_seed = int(seed) + offset
        splits, audit = _build(candidate_seed, offset)
        objective = float(audit["split_objective"])
        if objective < best_objective:
            best_splits = splits
            best_audit = audit
            best_objective = objective
    return best_splits, best_audit


def _data_fingerprint(data_dir: Path, domain: str) -> str:
    def _sample_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
        size = path.stat().st_size
        starts = [0]
        if size > chunk_size:
            starts.append(max(0, size // 2 - chunk_size // 2))
            starts.append(max(0, size - chunk_size))
        h = hashlib.sha256()
        with path.open("rb") as f:
            for start in sorted(set(starts)):
                f.seek(start)
                h.update(f.read(min(chunk_size, max(0, size - start))))
        return h.hexdigest()

    parts = []
    for suffix in ["X", "y"]:
        p = data_dir / f"{domain}_{suffix}.npy"
        st = p.stat()
        arr = np.load(p, mmap_mode="r")
        shape = "x".join(str(x) for x in arr.shape)
        dtype = str(arr.dtype)
        del arr
        parts.append(f"{p.name}:{st.st_size}:{shape}:{dtype}:{_sample_sha256(p)}")
    return "|".join(parts)


def _split_audit_valid(audit: dict, seed: int) -> bool:
    try:
        return (
            audit.get("policy") == SPLIT_POLICY
            and audit.get("split_policy", audit.get("policy")) == SPLIT_POLICY
            and audit.get("split_algo_version") == SPLIT_ALGO_VERSION
            and audit.get("split_assignment_mode_family") == SPLIT_ASSIGNMENT_MODE
            and audit.get("split_objective_version") == SPLIT_OBJECTIVE_VERSION
            and int(audit.get("candidate_seeds_tried")) == int(SPLIT_CANDIDATE_SEEDS)
            and int(audit.get("seed")) == int(seed)
        )
    except Exception:
        return False


def _split_file_valid(path: Path, n_rows: int, seed: int, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path, allow_pickle=False)
        required = {
            "train_idx", "val_idx", "test_idx", "row_group_id",
            "seed", "policy", "split_policy", "split_algo_version",
            "split_assignment_mode_family", "split_objective_version",
            "candidate_seeds_tried", "chosen_seed", "data_fingerprint",
        }
        if not required.issubset(set(data.files)):
            return False
        if str(np.asarray(data["policy"]).item()) != SPLIT_POLICY:
            return False
        if str(np.asarray(data["split_policy"]).item()) != SPLIT_POLICY:
            return False
        if str(np.asarray(data["split_algo_version"]).item()) != SPLIT_ALGO_VERSION:
            return False
        if str(np.asarray(data["split_assignment_mode_family"]).item()) != SPLIT_ASSIGNMENT_MODE:
            return False
        if str(np.asarray(data["split_objective_version"]).item()) != SPLIT_OBJECTIVE_VERSION:
            return False
        if int(np.asarray(data["candidate_seeds_tried"]).item()) != int(SPLIT_CANDIDATE_SEEDS):
            return False
        if int(np.asarray(data["seed"]).item()) != int(seed):
            return False
        if str(np.asarray(data["data_fingerprint"]).item()) != fingerprint:
            return False
        if len(data["row_group_id"]) != n_rows:
            return False
        total = len(data["train_idx"]) + len(data["val_idx"]) + len(data["test_idx"])
        return total == n_rows
    except Exception:
        return False


def ensure_duplicate_aware_splits(
    data_dir: str | Path,
    seed: int = LODO_SPLIT_SEED,
    domain_arrays: Optional[dict] = None,
    force: bool = False,
) -> dict:
    """Create or validate persisted duplicate-aware split files for all domains."""
    data_dir = Path(data_dir)
    split_dir = data_dir / SPLIT_DIRNAME
    split_dir.mkdir(parents=True, exist_ok=True)
    audit_path = split_dir / "split_audit.json"

    audit = {
        "policy": SPLIT_POLICY,
        "split_policy": SPLIT_POLICY,
        "split_algo_version": SPLIT_ALGO_VERSION,
        "split_assignment_mode_family": SPLIT_ASSIGNMENT_MODE,
        "split_objective_version": SPLIT_OBJECTIVE_VERSION,
        "candidate_seeds_tried": int(SPLIT_CANDIDATE_SEEDS),
        "seed": int(seed),
        "domains": {},
    }

    for dom in ALL_DOMAINS:
        if domain_arrays is not None and dom in domain_arrays:
            X, y = domain_arrays[dom]
        else:
            X = np.load(data_dir / f"{dom}_X.npy", mmap_mode="r")
            y = np.load(data_dir / f"{dom}_y.npy", mmap_mode="r")
        y = np.asarray(y, dtype=np.int64)
        split_path = split_dir / f"{dom}_split.npz"
        domain_audit_path = split_dir / f"{dom}_split_audit.json"
        fingerprint = _data_fingerprint(data_dir, dom)

        if not force and _split_file_valid(split_path, len(y), seed, fingerprint) and domain_audit_path.exists():
            try:
                with domain_audit_path.open(encoding="utf-8") as f:
                    dom_audit = json.load(f)
                if _split_audit_valid(dom_audit, seed):
                    audit["domains"][dom] = dom_audit
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        splits, dom_audit = _split_indices_from_groups(np.asarray(X), y, seed)
        np.savez_compressed(
            split_path,
            train_idx=splits["train_idx"],
            val_idx=splits["val_idx"],
            test_idx=splits["test_idx"],
            row_group_id=splits["row_group_id"],
            seed=np.asarray(seed, dtype=np.int64),
            policy=np.asarray(SPLIT_POLICY),
            split_policy=np.asarray(SPLIT_POLICY),
            split_algo_version=np.asarray(SPLIT_ALGO_VERSION),
            split_assignment_mode_family=np.asarray(SPLIT_ASSIGNMENT_MODE),
            split_assignment_mode=np.asarray(dom_audit.get("split_assignment_mode", "")),
            split_objective_version=np.asarray(SPLIT_OBJECTIVE_VERSION),
            candidate_seeds_tried=np.asarray(SPLIT_CANDIDATE_SEEDS, dtype=np.int64),
            chosen_seed=np.asarray(dom_audit.get("chosen_seed", seed), dtype=np.int64),
            objective_value=np.asarray(dom_audit.get("objective_value", dom_audit.get("split_objective", float("nan")))),
            max_attack_ratio_deviation=np.asarray(dom_audit.get("max_attack_ratio_deviation", float("nan"))),
            data_fingerprint=np.asarray(fingerprint),
        )
        _atomic_write_json_if_changed(domain_audit_path, dom_audit)
        audit["domains"][dom] = dom_audit

    _atomic_write_json_if_changed(audit_path, audit)
    return audit










def canonicalize_attack_family_name(name: object) -> str:
    """Normalize attack-family strings for shared multitask labels."""
    if name is None:
        return "unknown"
    text = str(name).strip()
    if not text:
        return "unknown"
    return " ".join(text.split()).lower()


def build_attack_family_vocab(attacks_by_domain: dict[str, np.ndarray]) -> tuple[dict[str, int], list[str]]:
    """Build a stable union vocabulary from source-domain attack-family arrays."""
    labels = set()
    for attack_names in attacks_by_domain.values():
        for name in attack_names:
            labels.add(canonicalize_attack_family_name(name))
    ordered = sorted(labels)
    vocab = {name: idx for idx, name in enumerate(ordered)}
    return vocab, ordered

class LODODataset(Dataset):
    """PyTorch dataset wrapper for one LODO domain slice.

    Each sample returns (X, y, domain_label) and optionally attack_family.
    domain_label is 0/1/2 for source domains in the LODO train mix.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, domain_label: int,
                 attack_family: np.ndarray | None = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.domain_label = domain_label
        self.attack_family = None
        if attack_family is not None:
            self.attack_family = torch.from_numpy(np.asarray(attack_family, dtype=np.int64))

        assert len(self.X) == len(self.y), \
            f"X ({len(self.X)})  y ({len(self.y)}) "
        if self.attack_family is not None:
            assert len(self.attack_family) == len(self.y), \
                f"attack_family ({len(self.attack_family)})  y ({len(self.y)}) "

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.attack_family is not None:
            return self.X[idx], self.y[idx], self.domain_label, self.attack_family[idx]
        return self.X[idx], self.y[idx], self.domain_label


# ============================================================
# ============================================================

class LODODataLoader:
    """Leave-one-domain-out data loader for the NFv2 NIDS benchmark.

    Example::

        loader = LODODataLoader(data_dir='./DATA/nfv2_processed', target_domain='unsw')
        loader.prepare()
        train_datasets = loader.get_train_datasets()
        X_in, y_in = loader.get_in_domain_eval()
        X_cross, y_cross = loader.get_cross_domain_eval()
    """

    def __init__(self, data_dir: str, target_domain: str, seed: int = LODO_SPLIT_SEED,
                 use_invalid_indicators: bool = True,
                 target_few_shot_n: int = 0,
                 target_few_shot_seed: int = 0,
                 target_few_shot_balanced: bool = True,
                 target_few_shot_policy: str = DEFAULT_FEW_SHOT_POLICY,
                 attack_family_policy: str = "exact",
                 source_domains: Optional[Iterable[str]] = None):
        assert target_domain in ALL_DOMAINS, (
            f"unknown target_domain {target_domain!r}; expected one of "
            f"{ALL_DOMAINS}"
        )

        self.data_dir = Path(data_dir)
        self.target_domain = target_domain
        if source_domains is None:
            self.source_domains = [d for d in ALL_DOMAINS if d != target_domain]
            self._restricted_source_domains = False
        else:
            selected = list(source_domains)
            if not selected:
                raise ValueError("source_domains must contain at least one domain")
            if len(set(selected)) != len(selected):
                raise ValueError(f"source_domains contains duplicates: {selected}")
            unknown = [domain for domain in selected if domain not in ALL_DOMAINS]
            if unknown:
                raise ValueError(
                    f"unknown source domains {unknown}; expected members of {ALL_DOMAINS}"
                )
            if target_domain in selected:
                raise ValueError(
                    f"target_domain {target_domain!r} cannot also be a source domain"
                )
            self.source_domains = selected
            self._restricted_source_domains = True
        self.seed = seed
        self.use_invalid_indicators = bool(use_invalid_indicators)
        self.target_few_shot_n = int(target_few_shot_n)
        self.target_few_shot_seed = int(target_few_shot_seed)
        self.target_few_shot_balanced = bool(target_few_shot_balanced)
        _validate_few_shot_policy_balance(
            target_few_shot_policy,
            self.target_few_shot_balanced,
        )
        self.target_few_shot_policy = target_few_shot_policy
        if attack_family_policy not in SUPPORTED_ATTACK_FAMILY_POLICIES:
            raise ValueError(
                f"unsupported attack_family_policy={attack_family_policy!r}; "
                f"expected {sorted(SUPPORTED_ATTACK_FAMILY_POLICIES)}"
            )
        self.attack_family_policy = attack_family_policy

        self._prepared = False
        self.scaler = None
        self._train = {}   # domain -> (X, y)
        self._val = {}
        self._test_in = {}
        self._target_X = None
        self._target_y = None
        self._source_train_unscaled = {}
        self._source_val_unscaled = {}
        self._source_test_in_unscaled = {}
        self._source_train_attack_family = {}
        self._source_val_attack_family = {}
        self._source_test_in_attack_family = {}
        self._target_attack_family = None
        self._target_X_unscaled = None
        self._target_duplicate_group_id = None
        self._target_support_candidate_mask = None
        self._target_fixed_eval_mask = None
        self._few_shot_idx = None
        self._few_shot_group_ids = None
        self._few_shot_class_ranks = None
        self._few_shot_train_mask = None
        self._few_shot_val_mask = None
        self._few_shot_X = None
        self._few_shot_y = None
        self._few_shot_attack_family = None
        self._cross_eval_mask = None
        self.attack_family_vocab = {}
        self.attack_family_labels = []
        self.feature_names = None
        self.preprocessing_audit = {}
        self.split_audit = {}

    def prepare(self):
        """Load arrays, build splits, fit source-only preprocessing, transform all splits."""
        if self._prepared:
            return

        feature_names_path = self.data_dir / "feature_names.txt"
        if feature_names_path.exists():
            self.feature_names = [
                line.strip()
                for line in feature_names_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        all_data = {}
        all_attack = {}
        required_domains = list(self.source_domains) + [self.target_domain]
        for dom in required_domains:
            X = np.load(self.data_dir / f"{dom}_X.npy")
            y = np.load(self.data_dir / f"{dom}_y.npy")
            all_data[dom] = (X, y)
            attack_path = self.data_dir / f"{dom}_attack.npy"
            if attack_path.exists():
                attack = np.load(attack_path, allow_pickle=True)
                if len(attack) != len(y):
                    raise ValueError(
                        f"attack metadata length mismatch for {dom}: "
                        f"attack={len(attack)} vs y={len(y)}"
                    )
                if self.attack_family_policy == "exact":
                    mapper = canonicalize_attack_family_name
                elif self.attack_family_policy == "coarse13_v1":
                    mapper = coarse_attack_family
                else:
                    mapper = shared4_attack_family
                all_attack[dom] = np.asarray([mapper(x) for x in attack], dtype=object)

        if self._restricted_source_domains:
            domain_audits = {}
            for dom in required_domains:
                split_path = self.data_dir / SPLIT_DIRNAME / f"{dom}_split.npz"
                audit_path = self.data_dir / SPLIT_DIRNAME / f"{dom}_split_audit.json"
                if not split_path.exists() or not audit_path.exists():
                    raise FileNotFoundError(
                        "restricted-source loading requires existing split and audit "
                        f"artifacts for {dom}: {split_path}, {audit_path}"
                    )
                with audit_path.open(encoding="utf-8") as handle:
                    domain_audits[dom] = json.load(handle)
            self.split_audit = {
                "policy": SPLIT_POLICY,
                "seed": int(self.seed),
                "restricted_source_domains": list(self.source_domains),
                "target_domain": self.target_domain,
                "domains": domain_audits,
            }
        else:
            self.split_audit = ensure_duplicate_aware_splits(
                self.data_dir, seed=self.seed, domain_arrays=all_data
            )

        source_attack = {dom: all_attack[dom] for dom in self.source_domains if dom in all_attack}
        if len(source_attack) == len(self.source_domains):
            self.attack_family_vocab, self.attack_family_labels = build_attack_family_vocab(source_attack)

        for dom in self.source_domains:
            X, y = all_data[dom]
            split = load_domain_split(self.data_dir, dom)
            tr_idx = split["train_idx"]
            val_idx = split["val_idx"]
            test_idx = split["test_idx"]
            self._source_train_unscaled[dom] = (X[tr_idx], y[tr_idx])
            self._source_val_unscaled[dom] = (X[val_idx], y[val_idx])
            self._source_test_in_unscaled[dom] = (X[test_idx], y[test_idx])
            if dom in all_attack and self.attack_family_vocab:
                attack = all_attack[dom]
                self._source_train_attack_family[dom] = np.asarray(
                    [self.attack_family_vocab[name] for name in attack[tr_idx]],
                    dtype=np.int64,
                )
                self._source_val_attack_family[dom] = np.asarray(
                    [self.attack_family_vocab[name] for name in attack[val_idx]],
                    dtype=np.int64,
                )
                self._source_test_in_attack_family[dom] = np.asarray(
                    [self.attack_family_vocab[name] for name in attack[test_idx]],
                    dtype=np.int64,
                )

        self._target_X, self._target_y = all_data[self.target_domain]
        self._target_X_unscaled = self._target_X
        target_split = load_domain_split(self.data_dir, self.target_domain)
        self._target_duplicate_group_id = target_split["row_group_id"]
        self._target_support_candidate_mask = np.zeros(len(self._target_y), dtype=bool)
        self._target_support_candidate_mask[target_split["train_idx"]] = True
        self._target_fixed_eval_mask = np.zeros(len(self._target_y), dtype=bool)
        self._target_fixed_eval_mask[target_split["test_idx"]] = True
        if self.target_domain in all_attack and self.attack_family_vocab:
            self._target_attack_family = np.asarray(
                [self.attack_family_vocab.get(name, -1) for name in all_attack[self.target_domain]],
                dtype=np.int64,
            )

        X_concat_raw = np.concatenate([self._source_train_unscaled[d][0] for d in self.source_domains])
        prep = fit_source_train_preprocessor(X_concat_raw, feature_names=self.feature_names)
        X_concat_capped, X_concat_ind = apply_source_train_preprocessor(X_concat_raw, prep)
        self.scaler = StandardScaler()
        self.scaler.fit(X_concat_capped)
        self.preprocessing_audit = {
            "policy": FEATURE_PREPROCESS_POLICY,
            "source_domains": self.source_domains,
            "restricted_source_domains": bool(self._restricted_source_domains),
            "target_domain": self.target_domain,
            "invalid_sentinel_abs": INVALID_SENTINEL_ABS,
            "cap_lower_quantile": CAP_LOWER_Q,
            "cap_upper_quantile": CAP_UPPER_Q,
            "use_invalid_indicators": self.use_invalid_indicators,
            "source_train_invalid_rate_mean": float(np.mean(prep["invalid_rates"])),
            "source_train_invalid_rate_max": float(np.max(prep["invalid_rates"])),
        }
        del X_concat_raw, X_concat_capped, X_concat_ind  # 

        def _transform(X):
            Xc, invalid = apply_source_train_preprocessor(X, prep)
            Xt = self.scaler.transform(Xc).astype(np.float32)
            np.clip(Xt, -SCALE_CLIP, SCALE_CLIP, out=Xt)
            if self.use_invalid_indicators:
                return np.concatenate([Xt, invalid], axis=1).astype(np.float32, copy=False)
            return Xt.astype(np.float32, copy=False)

        for dom in self.source_domains:
            Xtr, ytr = self._source_train_unscaled[dom]
            Xval, yval = self._source_val_unscaled[dom]
            Xtest_in, ytest_in = self._source_test_in_unscaled[dom]
            self._train[dom] = (_transform(Xtr), ytr)
            self._val[dom] = (_transform(Xval), yval)
            self._test_in[dom] = (_transform(Xtest_in), ytest_in)

        self._target_X = _transform(self._target_X)

        if self.feature_names is not None and self.use_invalid_indicators:
            self.feature_names = (
                list(self.feature_names)
                + [f"{name}_invalid_or_sentinel" for name in self.feature_names]
            )

        self._prepared = True
        if self.target_few_shot_n > 0:
            self.configure_few_shot_split(
                n=self.target_few_shot_n,
                seed=self.target_few_shot_seed,
                balanced=self.target_few_shot_balanced,
                policy=self.target_few_shot_policy,
            )

    def configure_few_shot_split(self, n: int, seed: int,
                                 balanced: bool = True,
                                 policy: str = DEFAULT_FEW_SHOT_POLICY):
        """Materialize a target support split without repeating preprocessing."""
        assert self._prepared, "call prepare() before configuring few-shot support"
        _validate_few_shot_policy_balance(policy, bool(balanced))
        self.target_few_shot_n = int(n)
        self.target_few_shot_seed = int(seed)
        self.target_few_shot_balanced = bool(balanced)
        self.target_few_shot_policy = policy
        self._materialize_few_shot_split()

    def _materialize_few_shot_split(self):
        """Carve a stratified subset of target rows for few-shot DA training.

        The selected rows are also masked out of the cross-domain evaluation
        set to preserve evaluation integrity. Selection is deterministic in
        ``target_few_shot_seed`` so that the same N rows are chosen across
        runs (frozen split).

        Stores:
          self._few_shot_X / _y / _attack_family
          self._few_shot_idx           # absolute indices into self._target_X
          self._cross_eval_mask        # bool[len(target)], True = keep in eval
        """
        n_total = int(self._target_y.shape[0])
        n = self.target_few_shot_n
        if n <= 0:
            raise ValueError(f"target_few_shot_n must be positive, got {n}")

        if self.target_few_shot_policy in GROUP_AWARE_FEW_SHOT_POLICIES:
            split = build_group_nested_few_shot_split(
                self._target_y,
                self._target_duplicate_group_id,
                n=n,
                seed=self.target_few_shot_seed,
                balanced=self.target_few_shot_balanced,
                candidate_mask=self._target_support_candidate_mask,
                evaluation_mask=self._target_fixed_eval_mask,
                policy=self.target_few_shot_policy,
            )
            sel = split["selected_indices"]
            mask = split["eval_mask"]
            self._few_shot_group_ids = split["selected_group_ids"]
            self._few_shot_class_ranks = split["class_ranks"]
            self._few_shot_train_mask = split["train_mask"]
            self._few_shot_val_mask = split["val_mask"]
        else:
            rng = np.random.default_rng(self.target_few_shot_seed)
            if self.target_few_shot_balanced:
                pos = np.flatnonzero(self._target_y == 1)
                neg = np.flatnonzero(self._target_y == 0)
                n_pos_want = min(n // 2, pos.shape[0])
                n_neg_want = min(n - n_pos_want, neg.shape[0])
                n_pos_want = min(n - n_neg_want, pos.shape[0])
                sel_pos = rng.choice(pos, size=n_pos_want, replace=False)
                sel_neg = rng.choice(neg, size=n_neg_want, replace=False)
                sel = np.concatenate([sel_pos, sel_neg]).astype(np.int64)
            else:
                sel = rng.choice(n_total, size=min(n, n_total), replace=False).astype(np.int64)
            sel.sort()
            mask = np.ones(n_total, dtype=bool)
            mask[sel] = False
            self._few_shot_group_ids = self._target_duplicate_group_id[sel].copy()
            self._few_shot_class_ranks = None
            self._few_shot_train_mask = None
            self._few_shot_val_mask = None

        self._few_shot_idx = sel
        self._cross_eval_mask = mask
        self._few_shot_X = self._target_X[sel].copy()
        self._few_shot_y = self._target_y[sel].copy()
        if self._target_attack_family is not None:
            self._few_shot_attack_family = self._target_attack_family[sel].copy()
        else:
            self._few_shot_attack_family = None

    def get_few_shot_dataset(self, domain_label: int,
                              include_attack_family: bool = False,
                              subset: str = "all"):
        """Return all, fit, or validation representatives as an LODODataset."""
        assert self._prepared
        if self.target_few_shot_n <= 0 or self._few_shot_X is None:
            return None
        if subset not in {"all", "train", "val"}:
            raise ValueError(f"unsupported few-shot subset={subset!r}")
        if subset == "all":
            subset_mask = np.ones(len(self._few_shot_y), dtype=bool)
        else:
            partition = (
                self._few_shot_train_mask if subset == "train"
                else self._few_shot_val_mask
            )
            if partition is None:
                raise ValueError(
                    f"few-shot subset {subset!r} is unavailable for legacy policy "
                    f"{self.target_few_shot_policy!r}"
                )
            subset_mask = partition
        attack_family = None
        if include_attack_family:
            if self._few_shot_attack_family is None:
                raise ValueError(
                    "attack-family metadata unavailable for target few-shot subset"
                )
            attack_family = self._few_shot_attack_family[subset_mask]
        return LODODataset(
            self._few_shot_X[subset_mask], self._few_shot_y[subset_mask],
            domain_label=domain_label,
            attack_family=attack_family,
        )

    def get_few_shot_validation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the held-out labelled support representatives used for selection."""
        assert self._prepared
        if self._few_shot_val_mask is None:
            raise ValueError("few-shot validation requires the group-nested policy")
        return (
            self._few_shot_X[self._few_shot_val_mask],
            self._few_shot_y[self._few_shot_val_mask],
        )

    def get_cross_domain_eval_excluding_few_shot(self):
        """Cross-domain eval (X, y) with the few-shot rows removed.

        Falls back to the full cross-domain set if no few-shot subset has
        been materialized.
        """
        assert self._prepared
        if self._cross_eval_mask is None:
            return self._target_X, self._target_y
        return self._target_X[self._cross_eval_mask], self._target_y[self._cross_eval_mask]

    def get_cross_domain_eval_unscaled_excluding_few_shot(self):
        """Same as get_cross_domain_eval_excluding_few_shot but on raw arrays."""
        assert self._prepared
        if self._cross_eval_mask is None:
            return self._target_X_unscaled, self._target_y
        return self._target_X_unscaled[self._cross_eval_mask], self._target_y[self._cross_eval_mask]

    def get_few_shot_audit(self) -> dict:
        """Metadata describing the few-shot subset (for artifact persistence)."""
        if self._few_shot_idx is None:
            return {
                "schema_version": 2,
                "target_few_shot_n": 0,
                "target_few_shot_policy": self.target_few_shot_policy,
            }
        train_mask = self._few_shot_train_mask
        val_mask = self._few_shot_val_mask
        selected_group_ids = np.asarray(self._few_shot_group_ids)
        group_bytes = np.ascontiguousarray(selected_group_ids).view(np.uint8)
        idx_bytes = np.ascontiguousarray(self._few_shot_idx).view(np.uint8)
        eval_groups = np.unique(self._target_duplicate_group_id[self._cross_eval_mask])
        overlap = int(np.intersect1d(eval_groups, selected_group_ids).size)
        return {
            "schema_version": 2,
            "target_few_shot_n": int(self.target_few_shot_n),
            "target_few_shot_seed": int(self.target_few_shot_seed),
            "target_few_shot_balanced": bool(self.target_few_shot_balanced),
            "target_few_shot_policy": self.target_few_shot_policy,
            "support_sampling_design": (
                "class_balanced_duplicate_groups"
                if self.target_few_shot_policy == FEW_SHOT_POLICY_GROUP_NESTED
                else (
                    "uniform_candidate_duplicate_groups"
                    if self.target_few_shot_policy == FEW_SHOT_POLICY_GROUP_RANDOM_NESTED
                    else "legacy_rows"
                )
            ),
            "support_selection_uses_labels": (
                bool(self.target_few_shot_balanced)
            ),
            "validation_assignment": (
                f"per_class_rank_every_{FEW_SHOT_VAL_EVERY}"
                if self.target_few_shot_policy == FEW_SHOT_POLICY_GROUP_NESTED
                else (
                    f"global_acquisition_rank_every_{FEW_SHOT_VAL_EVERY}"
                    if self.target_few_shot_policy == FEW_SHOT_POLICY_GROUP_RANDOM_NESTED
                    else None
                )
            ),
            "n_unique_groups_selected": int(len(np.unique(selected_group_ids))),
            "n_pos": int((self._few_shot_y == 1).sum()),
            "n_neg": int((self._few_shot_y == 0).sum()),
            "n_fit": int(train_mask.sum()) if train_mask is not None else None,
            "n_val": int(val_mask.sum()) if val_mask is not None else None,
            "fit_pos": int(self._few_shot_y[train_mask].sum()) if train_mask is not None else None,
            "val_pos": int(self._few_shot_y[val_mask].sum()) if val_mask is not None else None,
            "selected_indices_first10": self._few_shot_idx[:10].tolist(),
            "selected_indices_count": int(self._few_shot_idx.shape[0]),
            "selected_indices_sha256": hashlib.sha256(idx_bytes).hexdigest(),
            "selected_group_ids_sha256": hashlib.sha256(group_bytes).hexdigest(),
            "n_target_total": int(self._target_y.shape[0]),
            "n_eval_after_exclude": int(self._cross_eval_mask.sum()),
            "eval_attack_ratio": float(self._target_y[self._cross_eval_mask].mean()),
            "support_eval_group_overlap": overlap,
            "evaluation_scope": (
                "fixed_target_internal_test"
                if self.target_few_shot_policy in GROUP_AWARE_FEW_SHOT_POLICIES
                else "full_target_excluding_selected_rows"
            ),
        }

    def get_train_datasets(self, include_attack_family: bool = False) -> list:
        """ 3  Dataset ,domain_label  0, 1, 2."""
        assert self._prepared, " prepare()"
        out = []
        for i, dom in enumerate(self.source_domains):
            X, y = self._train[dom]
            attack_family = None
            if include_attack_family:
                attack_family = self._source_train_attack_family.get(dom)
                if attack_family is None:
                    raise ValueError(
                        f"attack-family metadata unavailable for source domain {dom}"
                    )
            out.append(LODODataset(X, y, domain_label=i, attack_family=attack_family))
        return out

    def get_proxy_lodo_split(self, proxy_source: str,
                            include_attack_family: bool = False) -> dict:
        """Treat ``proxy_source`` (one of the LODO source domains) as a held-out
        pseudo-target and return the remaining sources plus the proxy data.

        Returned dict has keys:
            train_datasets: list[LODODataset] for the non-proxy sources, with
                consecutive domain_label = 0..(n-1)
            val_per_domain: dict[domain -> (X, y)] using only non-proxy source
                validation splits, for early stopping and HP selection
            proxy_train_X, proxy_train_y: proxy source's train rows (used for
                cross-style proxy evaluation; analogous to target eval)
            proxy_val_X, proxy_val_y: proxy source's val rows
            proxy_test_X, proxy_test_y: proxy source's internal-test rows
            non_proxy_sources: list of domains kept for training
        """
        assert self._prepared, " prepare()"
        if proxy_source not in self.source_domains:
            raise ValueError(
                f"proxy_source={proxy_source!r} not in current source_domains "
                f"{self.source_domains}; the LODO target is {self.target_domain!r}."
            )
        non_proxy = [d for d in self.source_domains if d != proxy_source]
        train_datasets = []
        for i, dom in enumerate(non_proxy):
            X, y = self._train[dom]
            attack_family = None
            if include_attack_family:
                attack_family = self._source_train_attack_family.get(dom)
                if attack_family is None:
                    raise ValueError(f"attack-family metadata unavailable for proxy source {dom}")
            train_datasets.append(LODODataset(X, y, domain_label=i, attack_family=attack_family))

        val_per_domain = {d: self._val[d] for d in non_proxy}
        proxy_train_X, proxy_train_y = self._train[proxy_source]
        proxy_val_X, proxy_val_y = self._val[proxy_source]
        proxy_test_X, proxy_test_y = self._test_in[proxy_source]

        return {
            "train_datasets": train_datasets,
            "val_per_domain": val_per_domain,
            "proxy_train_X": proxy_train_X,
            "proxy_train_y": proxy_train_y,
            "proxy_val_X": proxy_val_X,
            "proxy_val_y": proxy_val_y,
            "proxy_test_X": proxy_test_X,
            "proxy_test_y": proxy_test_y,
            "non_proxy_sources": list(non_proxy),
            "proxy_source": proxy_source,
        }

    def get_source_train_attack_family(self, per_domain: bool = False):
        """Return source-train attack-family ids aligned with transformed train splits."""
        assert self._prepared
        if not self._source_train_attack_family:
            raise ValueError("attack-family metadata was not loaded for the current dataset")
        if per_domain:
            return {d: self._source_train_attack_family[d] for d in self.source_domains}
        return np.concatenate([self._source_train_attack_family[d] for d in self.source_domains])

    def get_attack_family_info(self) -> dict:
        """Return shared source attack-family vocabulary metadata."""
        assert self._prepared
        return {
            "n_classes": int(len(self.attack_family_labels)),
            "labels": list(self.attack_family_labels),
            "vocab": dict(self.attack_family_vocab),
            "policy": self.attack_family_policy,
            "ontology_version": (
                SHARED4_ONTOLOGY_VERSION if self.attack_family_policy == "shared4_v1"
                else ONTOLOGY_VERSION if self.attack_family_policy == "coarse13_v1"
                else "exact_attack_string_v1"
            ),
        }

    def get_source_val(self, per_domain: bool = False):
        """
         3  val ().

        per_domain=True  dict: {domain: (X, y)};
         (X, y).
        """
        assert self._prepared
        if per_domain:
            return {d: self._val[d] for d in self.source_domains}
        X = np.concatenate([self._val[d][0] for d in self.source_domains])
        y = np.concatenate([self._val[d][1] for d in self.source_domains])
        return X, y

    def get_source_val_unscaled(self, per_domain: bool = False):
        """Return raw processed source-validation arrays before LODO transforms."""
        assert self._prepared
        if per_domain:
            return {d: self._source_val_unscaled[d] for d in self.source_domains}
        X = np.concatenate([self._source_val_unscaled[d][0] for d in self.source_domains])
        y = np.concatenate([self._source_val_unscaled[d][1] for d in self.source_domains])
        return X, y

    def get_source_train(self, per_domain: bool = False):
        """Return transformed source train arrays."""
        assert self._prepared
        if per_domain:
            return {d: self._train[d] for d in self.source_domains}
        X = np.concatenate([self._train[d][0] for d in self.source_domains])
        y = np.concatenate([self._train[d][1] for d in self.source_domains])
        return X, y

    def get_in_domain_eval(self):
        """Return concatenated source internal-test arrays for in-domain eval."""
        assert self._prepared
        X = np.concatenate([self._test_in[d][0] for d in self.source_domains])
        y = np.concatenate([self._test_in[d][1] for d in self.source_domains])
        return X, y

    def get_in_domain_eval_unscaled(self):
        """Return raw processed source internal-test arrays before LODO transforms."""
        assert self._prepared
        X = np.concatenate([self._source_test_in_unscaled[d][0] for d in self.source_domains])
        y = np.concatenate([self._source_test_in_unscaled[d][1] for d in self.source_domains])
        return X, y

    def get_in_domain_eval_per_domain(self) -> dict:
        """Return transformed source internal-test arrays per source domain."""
        assert self._prepared
        return {d: self._test_in[d] for d in self.source_domains}

    def get_cross_domain_eval(self):
        """Return the full cross-domain evaluation set (X, y) for the LODO target."""
        assert self._prepared
        return self._target_X, self._target_y

    def get_cross_domain_eval_unscaled(self):
        """Return raw processed target arrays before LODO capping/scaling."""
        assert self._prepared
        return self._target_X_unscaled, self._target_y

    def get_source_val_per_domain(self) -> dict:
        """Return per-source-domain validation arrays ``{dom: (X, y)}``."""
        assert self._prepared
        return {d: self._val[d] for d in self.source_domains}

    def get_split_audit(self) -> dict:
        assert self._prepared
        return self.split_audit

    @property
    def n_features(self) -> int:
        """Return the transformed feature dimension after ``prepare()``."""
        assert self._prepared
        return self._train[self.source_domains[0]][0].shape[1]

    def summary(self) -> dict:
        """Return a compact dataset summary after ``prepare()``."""
        assert self._prepared
        info = {
            "target_domain": self.target_domain,
            "source_domains": self.source_domains,
            "n_features": self.n_features,
            "n_attack_family_classes": int(len(self.attack_family_labels)),
        }
        for dom in self.source_domains:
            Xtr, ytr = self._train[dom]
            info[f"{dom}_train"] = {"n": len(ytr), "attack_ratio": float((ytr==1).mean())}
        info[f"{self.target_domain}_target"] = {
            "n": len(self._target_y),
            "attack_ratio": float((self._target_y==1).mean()),
        }
        return info


# ============================================================
# Domain-balanced batch sampler
# ============================================================

class DomainBalancedBatchSampler(Sampler):
    """Yield batches with equal counts from each source domain.

    Required by IRM, GroupDRO, CORAL, Fishr, BoDA, and DANN so that every
    batch covers all K source domains.
    """

    def __init__(self, domain_indices: dict, batch_size: int,
                 shuffle: bool = True, seed: int = 0):
        """
        :
            domain_indices: {domain_label (int): list of indices}
            batch_size:  len(domain_indices) 
        """
        self.domain_indices = {k: np.array(v) for k, v in domain_indices.items()}
        self.n_domains = len(domain_indices)
        self.batch_size = batch_size
        self.per_domain = batch_size // self.n_domains
        self.shuffle = shuffle
        self.seed = seed

        if self.batch_size % self.n_domains != 0:
            raise ValueError(
                f"batch_size ({batch_size})  ({self.n_domains}) "
            )

        min_size = min(len(v) for v in self.domain_indices.values())
        self.num_batches = min_size // self.per_domain

        self._epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)

        shuffled = {}
        for d, idx in self.domain_indices.items():
            if self.shuffle:
                perm = rng.permutation(len(idx))
                shuffled[d] = idx[perm]
            else:
                shuffled[d] = idx.copy()

        for b in range(self.num_batches):
            batch = []
            for d, idx_arr in shuffled.items():
                start = b * self.per_domain
                end = start + self.per_domain
                batch.extend(idx_arr[start:end].tolist())

            if self.shuffle:
                rng.shuffle(batch)

            yield batch

        self._epoch += 1

    def __len__(self):
        return self.num_batches


# ============================================================
# ============================================================

def build_train_loader(train_datasets: list, batch_size: int,
                       domain_balanced: bool = False,
                       num_workers: int = 2,
                       seed: int = 0):
    """
     DataLoader.

    :
        train_datasets: LODODataset (3 )
        batch_size: 
        domain_balanced: True ->  DomainBalancedBatchSampler;False ->  shuffle
        num_workers: DataLoader worker 

    :
        torch.utils.data.DataLoader
    """
    from torch.utils.data import ConcatDataset, DataLoader

    concat = ConcatDataset(train_datasets)

    if domain_balanced:
        domain_indices = {}
        offset = 0
        for ds in train_datasets:
            dl = ds.domain_label
            idx_range = list(range(offset, offset + len(ds)))
            domain_indices.setdefault(dl, []).extend(idx_range)
            offset += len(ds)

        sampler = DomainBalancedBatchSampler(
            domain_indices=domain_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        loader = DataLoader(
            concat,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            concat,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader
    
# ============================================================
# ============================================================

class GPUResidentBatchIterator:
    """
     GPU,batch  GPU .
     CPU↔GPU  +  DataLoader ,GPU  70%+.

    : MLP( <1M)+ ( <1M ).
     DataLoader + num_workers  GPU  5-20%,
     Python  + H2D  GPU .
    """
    def __init__(self, train_datasets, batch_size: int,
                 domain_balanced: bool = False,
                 device: str = 'cuda', seed: int = 0):
        import torch

        self.device = device
        self.batch_size = batch_size
        self.domain_balanced = domain_balanced
        self.seed = seed
        self._epoch = 0
        self.n_domains = len(train_datasets)
        family_flags = [getattr(ds, "attack_family", None) is not None for ds in train_datasets]
        if any(family_flags) and not all(family_flags):
            raise ValueError("either every train dataset must provide attack_family or none of them")
        self.has_attack_family = all(family_flags)

        X_list, y_list, d_list = [], [], []
        attack_family_list = []
        for ds in train_datasets:
            X_list.append(ds.X)
            y_list.append(ds.y)
            d_list.append(torch.full((len(ds),), ds.domain_label, dtype=torch.int64))
            if self.has_attack_family:
                attack_family_list.append(ds.attack_family)

        self.X = torch.cat(X_list).to(device, non_blocking=True)
        self.y = torch.cat(y_list).to(device, non_blocking=True)
        self.d = torch.cat(d_list).to(device, non_blocking=True)
        self.attack_family = None
        if self.has_attack_family:
            self.attack_family = torch.cat(attack_family_list).to(device, non_blocking=True)
        self.n = len(self.X)

        if domain_balanced:
            if batch_size % self.n_domains != 0:
                raise ValueError(
                    f"batch_size ({batch_size})  ({self.n_domains}) "
                )
            self.per_domain = batch_size // self.n_domains
            self.domain_indices = {}
            offset = 0
            for ds in train_datasets:
                self.domain_indices[ds.domain_label] = torch.arange(
                    offset, offset + len(ds), device=device
                )
                offset += len(ds)
            min_size = min(len(v) for v in self.domain_indices.values())
            self.num_batches = min_size // self.per_domain
        else:
            self.num_batches = self.n // batch_size

    def __iter__(self):
        import torch
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + self._epoch)

        if self.domain_balanced:
            shuffled = {}
            for d, idx in self.domain_indices.items():
                perm = torch.randperm(len(idx), generator=gen, device=self.device)
                shuffled[d] = idx[perm]

            for b in range(self.num_batches):
                parts = []
                for d in sorted(shuffled.keys()):
                    start = b * self.per_domain
                    end = start + self.per_domain
                    parts.append(shuffled[d][start:end])
                batch_idx = torch.cat(parts)
                batch_perm = torch.randperm(len(batch_idx), generator=gen, device=self.device)
                batch_idx = batch_idx[batch_perm]
                if self.has_attack_family:
                    yield (
                        self.X[batch_idx],
                        self.y[batch_idx],
                        self.d[batch_idx],
                        self.attack_family[batch_idx],
                    )
                else:
                    yield self.X[batch_idx], self.y[batch_idx], self.d[batch_idx]
        else:
            perm = torch.randperm(self.n, generator=gen, device=self.device)
            for b in range(self.num_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                if self.has_attack_family:
                    yield self.X[idx], self.y[idx], self.d[idx], self.attack_family[idx]
                else:
                    yield self.X[idx], self.y[idx], self.d[idx]

        self._epoch += 1

    def __len__(self):
        return self.num_batches
