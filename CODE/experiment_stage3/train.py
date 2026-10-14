import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))
"""
Single-run LODO training entry point.

Wraps one (algorithm, target, hyperparameters, seed) configuration with
early stopping and optional artifact export.

Example:
    from CODE.experiment_stage3.train import train_one_run

    result = train_one_run(
        algorithm_name='erm',
        target_domain='unsw',
        hparams={'lr': 1e-4, 'batch_size': 512, 'dropout': 0.3, ...},
        seed=42,
        data_dir='./data/nfv2/processed',
        max_epochs=50,
        early_stop_patience=10,
        early_stop_metric='val_auroc_macro',
        device='cuda',
        verbose=True,
    )
"""
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

import numpy as np
import torch

from nidsbed.datasets import (
    DEFAULT_FEW_SHOT_POLICY,
    FEW_SHOT_POLICY_GROUP_NESTED,
    FEW_SHOT_POLICY_LEGACY,
    LODODataLoader,
    build_train_loader,
    unique_row_weights,
)
from nidsbed.algorithm_metadata import get_algorithm_metadata
from nidsbed.calibration import fit_temperature_from_logits, scores_from_logits
from nidsbed.lib.misc import seed_everything, get_class_weights, format_time
from nidsbed.model_selection import (
    evaluate_source_val_macro_avg,
    evaluate_full,
    predict_logits_batched,
)

def get_algorithm_class(name: str):
    """Return the algorithm class for the configured DG method."""
    name = name.lower()
    if name == 'erm':
        from nidsbed.algorithms.erm import ERM
        return ERM
    elif name == 'irm':
        from nidsbed.algorithms.irm import IRM
        return IRM
    elif name in ('coral', 'deepcoral'):
        from nidsbed.algorithms.coral import DeepCORAL
        return DeepCORAL
    elif name == 'groupdro':
        from nidsbed.algorithms.groupdro import GroupDRO
        return GroupDRO
    elif name == 'dann':
        from nidsbed.algorithms.dann import DANN
        return DANN
    elif name == 'fishr':
        from nidsbed.algorithms.fishr import Fishr
        return Fishr
    elif name == 'boda':
        from nidsbed.algorithms.boda import BoDA
        return BoDA
    elif name in ('multitask_erm', 'multitask_erm_shared4'):
        from nidsbed.algorithms.multitask_erm import MultitaskERM
        return MultitaskERM
    else:
        raise ValueError(
            f"unknown algorithm name: {name!r}. "
            f"Expected one of: erm, irm, coral (deepcoral), groupdro, "
            f"dann, fishr, boda, multitask_erm."
        )


def needs_domain_balanced_batch(name: str) -> bool:
    """Return whether the method needs domain-balanced batch sampling.

    IRM, GroupDRO, CORAL, Fishr, BoDA, and DANN aggregate per-domain
    statistics or penalties and therefore require each batch to cover all
    source domains. ERM and MultitaskERM use ordinary shuffled batches.
    """
    return name.lower() in [
        'irm', 'coral', 'deepcoral', 'groupdro',
        'dann', 'fishr', 'boda',
    ]


def _multiclass_inverse_freq_weights(y: np.ndarray, n_classes: int,
                                     clip_max: float = 20.0) -> np.ndarray:
    """Return mean-one inverse-frequency weights for multiclass auxiliary losses."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=int(n_classes)).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 1.0 / clip_max, clip_max)
    weights = weights * (len(weights) / np.sum(weights))
    return weights.astype(np.float32, copy=False)


def _build_algorithm_from_checkpoint_payload(ckpt: dict, device: str, class_weights_per_domain: dict):
    algo_name = ckpt['algorithm']
    AlgoCls = get_algorithm_class(algo_name)
    algorithm = AlgoCls(
        in_dim=ckpt['in_dim'],
        n_classes=ckpt['n_classes'],
        hparams=ckpt['hparams'],
        class_weights_per_domain=class_weights_per_domain,
        device=device,
    )
    algorithm.load_state_dict(ckpt['state_dict'])
    algorithm.eval()
    return algorithm


def save_source_intest_artifacts(algorithm, loader: LODODataLoader, device: str,
                                 save_dir: str) -> dict:
    """Persist predictions for the pooled source internal-test split."""
    import os

    os.makedirs(save_dir, exist_ok=True)
    eval_batch = int(os.environ.get('NIDS_EVAL_BATCH', 4096))
    X_in, y_in = loader.get_in_domain_eval()
    X_in_raw, _ = loader.get_in_domain_eval_unscaled()
    logits = predict_logits_batched(
        algorithm, X_in, batch_size=eval_batch, device=device
    ).astype(np.float32)
    scores = scores_from_logits(logits, temperature=1.0).astype(np.float32)
    weights = unique_row_weights(X_in_raw).astype(np.float32)

    np.save(os.path.join(save_dir, 'source_intest_logits.npy'), logits)
    np.save(os.path.join(save_dir, 'source_intest_scores.npy'), scores)
    np.save(os.path.join(save_dir, 'source_intest_labels.npy'), y_in.astype(np.int32))
    np.save(os.path.join(save_dir, 'source_intest_unique_weights.npy'), weights)
    return {
        'n_rows': int(len(y_in)),
        'n_unique_groups': float(np.sum(weights)),
    }


def save_eval_artifacts(algorithm, loader: LODODataLoader, train_datasets: list,
                        seed: int, device: str, save_dir: str,
                        algorithm_name: str, target_domain: str,
                        verbose: bool = True) -> dict:
    """Save downstream analysis artifacts from an already-trained model.

    Besides raw softmax scores, this writes source-validation temperature
    calibration artifacts. Downstream prior-shift analyses should use the
    calibrated scores by default.
    """
    import json
    import os

    os.makedirs(save_dir, exist_ok=True)
    eval_batch = int(os.environ.get('NIDS_EVAL_BATCH', 4096))

    X_cross, y_cross = loader.get_cross_domain_eval()
    target_logits = predict_logits_batched(
        algorithm, X_cross, batch_size=eval_batch, device=device
    ).astype(np.float32)
    target_scores = scores_from_logits(target_logits, temperature=1.0)
    np.save(os.path.join(save_dir, 'target_logits.npy'), target_logits)
    np.save(os.path.join(save_dir, 'target_scores.npy'), target_scores.astype(np.float32))
    np.save(os.path.join(save_dir, 'target_labels.npy'), y_cross.astype(np.int32))
    X_cross_raw, _ = loader.get_cross_domain_eval_unscaled()
    np.save(os.path.join(save_dir, 'target_unique_weights.npy'),
            unique_row_weights(X_cross_raw).astype(np.float32))

    if loader.target_few_shot_n > 0 and loader._cross_eval_mask is not None:
        np.save(os.path.join(save_dir, 'target_eval_mask.npy'),
                loader._cross_eval_mask.astype(bool))
        np.save(os.path.join(save_dir, 'target_few_shot_idx.npy'),
                loader._few_shot_idx.astype(np.int64))
        np.save(os.path.join(save_dir, 'target_few_shot_group_ids.npy'),
                np.asarray(loader._few_shot_group_ids))
        if loader._few_shot_train_mask is not None:
            np.save(os.path.join(save_dir, 'target_few_shot_train_mask.npy'),
                    loader._few_shot_train_mask.astype(bool))
            np.save(os.path.join(save_dir, 'target_few_shot_val_mask.npy'),
                    loader._few_shot_val_mask.astype(bool))
            np.savez_compressed(
                os.path.join(save_dir, 'support_split.npz'),
                selected_indices=loader._few_shot_idx.astype(np.int64),
                selected_group_ids=np.asarray(loader._few_shot_group_ids),
                class_ranks=loader._few_shot_class_ranks.astype(np.int64),
                train_mask=loader._few_shot_train_mask.astype(bool),
                val_mask=loader._few_shot_val_mask.astype(bool),
                eval_mask=loader._cross_eval_mask.astype(bool),
                policy=np.asarray(loader.target_few_shot_policy),
                support_seed=np.asarray(loader.target_few_shot_seed, dtype=np.int64),
            )
        # Also save eval-only versions (excluding few-shot training rows) so
        # downstream scripts can safely load without needing to apply the mask.
        eval_mask = loader._cross_eval_mask
        np.save(os.path.join(save_dir, 'target_scores_eval_only.npy'),
                target_scores[eval_mask].astype(np.float32))
        np.save(os.path.join(save_dir, 'target_labels_eval_only.npy'),
                y_cross[eval_mask].astype(np.int32))
        with open(os.path.join(save_dir, 'few_shot_audit.json'), 'w', encoding='utf-8') as f:
            json.dump(loader.get_few_shot_audit(), f, indent=2, ensure_ascii=False)

    X_val, y_val = loader.get_source_val()
    X_val_raw, _ = loader.get_source_val_unscaled()
    source_val_logits = predict_logits_batched(
        algorithm, X_val, batch_size=eval_batch, device=device
    ).astype(np.float32)
    source_val_scores = scores_from_logits(source_val_logits, temperature=1.0)
    temperature, calibration_meta = fit_temperature_from_logits(source_val_logits, y_val)
    source_val_scores_cal = scores_from_logits(source_val_logits, temperature=temperature)
    target_scores_cal = scores_from_logits(target_logits, temperature=temperature)

    np.save(os.path.join(save_dir, 'source_val_logits.npy'), source_val_logits)
    np.save(os.path.join(save_dir, 'source_val_scores.npy'), source_val_scores.astype(np.float32))
    np.save(os.path.join(save_dir, 'source_val_scores_calibrated.npy'),
            source_val_scores_cal.astype(np.float32))
    np.save(os.path.join(save_dir, 'source_val_labels.npy'), y_val.astype(np.int32))
    np.save(os.path.join(save_dir, 'source_val_unique_weights.npy'),
            unique_row_weights(X_val_raw).astype(np.float32))
    np.save(os.path.join(save_dir, 'target_scores_calibrated.npy'),
            target_scores_cal.astype(np.float32))

    save_source_intest_artifacts(
        algorithm=algorithm,
        loader=loader,
        device=device,
        save_dir=save_dir,
    )

    if loader.target_few_shot_n > 0 and loader._cross_eval_mask is not None:
        eval_mask = loader._cross_eval_mask
        np.save(os.path.join(save_dir, 'target_logits_eval_only.npy'),
                target_logits[eval_mask].astype(np.float32))
        np.save(os.path.join(save_dir, 'target_scores_calibrated_eval_only.npy'),
                target_scores_cal[eval_mask].astype(np.float32))
        np.save(os.path.join(save_dir, 'target_unique_weights_eval_only.npy'),
                unique_row_weights(X_cross_raw[eval_mask]).astype(np.float32))

    src_X_list, src_y_list = [], []
    for ds in train_datasets:
        src_X_list.append(ds.X.cpu().numpy() if hasattr(ds.X, 'cpu') else ds.X)
        src_y_list.append(ds.y.cpu().numpy() if hasattr(ds.y, 'cpu') else ds.y)
    src_X = np.concatenate(src_X_list, axis=0)
    src_y = np.concatenate(src_y_list, axis=0)

    rng_src = np.random.RandomState(seed)
    n_src = min(5000, len(src_y))
    src_idx = rng_src.choice(len(src_y), size=n_src, replace=False)
    src_X_sample = src_X[src_idx]
    src_y_sample = src_y[src_idx]

    source_logits = predict_logits_batched(
        algorithm, src_X_sample, batch_size=eval_batch, device=device
    ).astype(np.float32)
    source_scores = scores_from_logits(source_logits, temperature=1.0)
    np.save(os.path.join(save_dir, 'source_scores.npy'), source_scores.astype(np.float32))
    np.save(os.path.join(save_dir, 'source_labels.npy'), src_y_sample.astype(np.int32))
    np.save(os.path.join(save_dir, 'source_X_sample.npy'), src_X_sample.astype(np.float32))

    calibration_meta.update({
        'algorithm': algorithm_name,
        'target_domain': target_domain,
        'seed': int(seed),
        'calibration_split': 'concatenated_source_validation',
        'source_domains': loader.source_domains,
        'target_attack_rate': float(np.mean(y_cross)),
        'source_val_attack_rate': float(np.mean(y_val)),
        'score_variant_default': 'temperature_calibrated_source_val',
        'feature_preprocessing_policy': loader.preprocessing_audit.get('policy'),
        'split_policy': loader.split_audit.get('policy'),
    })
    with open(os.path.join(save_dir, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calibration_meta, f, indent=2, ensure_ascii=False)

    attack_family_info = loader.get_attack_family_info()
    if attack_family_info.get("n_classes", 0) > 0:
        with open(os.path.join(save_dir, 'attack_family_vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(attack_family_info, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  [save] calibration T={temperature:.4f} -> {save_dir}/")

    return calibration_meta


def backfill_calibration_artifacts(method: str, target_domain: str, seed: int,
                                   data_dir: str, device: str,
                                   save_dir: str, verbose: bool = True,
                                   use_invalid_indicators: bool = True) -> dict:
    """Rebuild score/calibration artifacts from an existing checkpoint.

    This lets --resume repair older partial runs after artifact requirements
    change, without rerunning model training when checkpoint.pt is valid.
    """
    import os

    ckpt_path = os.path.join(save_dir, 'checkpoint.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get('algorithm') != method or ckpt.get('target_domain') != target_domain or int(ckpt.get('seed')) != int(seed):
        raise ValueError(
            f"checkpoint metadata mismatch in {ckpt_path}: "
            f"algorithm={ckpt.get('algorithm')}, target={ckpt.get('target_domain')}, seed={ckpt.get('seed')}"
        )

    few_shot_audit = ckpt.get('target_few_shot_audit') or {}
    target_few_shot_n = int(ckpt.get(
        'target_few_shot_n',
        few_shot_audit.get('target_few_shot_n', 0),
    ))
    target_few_shot_seed = int(ckpt.get(
        'target_few_shot_seed',
        few_shot_audit.get('target_few_shot_seed', 0),
    ))
    target_few_shot_balanced = bool(ckpt.get(
        'target_few_shot_balanced',
        few_shot_audit.get('target_few_shot_balanced', True),
    ))
    target_few_shot_policy = ckpt.get(
        'target_few_shot_policy',
        few_shot_audit.get('target_few_shot_policy', FEW_SHOT_POLICY_LEGACY),
    )
    target_only = bool(ckpt.get('target_only', False))

    loader = LODODataLoader(
        data_dir=data_dir,
        target_domain=target_domain,
        use_invalid_indicators=use_invalid_indicators,
        target_few_shot_n=target_few_shot_n,
        target_few_shot_seed=target_few_shot_seed,
        target_few_shot_balanced=target_few_shot_balanced,
        target_few_shot_policy=target_few_shot_policy,
    )
    loader.prepare()

    use_attack_family = method.lower() == 'multitask_erm'
    train_datasets = loader.get_train_datasets(include_attack_family=use_attack_family)
    if target_few_shot_n > 0:
        if needs_domain_balanced_batch(method):
            raise NotImplementedError(
                f"few-shot backfill for domain-balanced algorithm {method!r} "
                "is not supported"
            )
        fs_domain_label = 0 if target_only else len(loader.source_domains)
        few_shot_ds = loader.get_few_shot_dataset(
            domain_label=fs_domain_label,
            include_attack_family=use_attack_family,
            subset=(
                "train"
                if target_few_shot_policy == FEW_SHOT_POLICY_GROUP_NESTED
                else "all"
            ),
        )
        if few_shot_ds is not None and len(few_shot_ds) > 0:
            if target_only and target_few_shot_policy == FEW_SHOT_POLICY_GROUP_NESTED:
                train_datasets = [few_shot_ds]
            elif target_only:
                from nidsbed.datasets import LODODataset
                fs_y_np = few_shot_ds.y.cpu().numpy()
                fs_X_np = few_shot_ds.X.cpu().numpy()
                fs_fam_np = (
                    few_shot_ds.attack_family.cpu().numpy()
                    if getattr(few_shot_ds, 'attack_family', None) is not None
                    else None
                )
                rng = np.random.default_rng(seed)
                val_mask = np.zeros(len(fs_y_np), dtype=bool)
                for cls in (0, 1):
                    cls_idx = np.where(fs_y_np == cls)[0]
                    if len(cls_idx) == 0:
                        continue
                    n_val_cls = max(1, int(round(len(cls_idx) * 0.2)))
                    val_pick = rng.choice(cls_idx, size=n_val_cls, replace=False)
                    val_mask[val_pick] = True
                train_mask = ~val_mask
                train_datasets = [
                    LODODataset(
                        fs_X_np[train_mask], fs_y_np[train_mask],
                        domain_label=fs_domain_label,
                        attack_family=(fs_fam_np[train_mask] if fs_fam_np is not None else None),
                    )
                ]
            else:
                train_datasets = list(train_datasets) + [few_shot_ds]

    class_weights_per_domain = {}
    for i, ds in enumerate(train_datasets):
        y_np = ds.y.numpy()
        class_weights_per_domain[i] = get_class_weights(y_np, clip_max=20.0, device=device)

    algorithm = _build_algorithm_from_checkpoint_payload(ckpt, device, class_weights_per_domain)
    return save_eval_artifacts(
        algorithm=algorithm,
        loader=loader,
        train_datasets=train_datasets,
        seed=seed,
        device=device,
        save_dir=save_dir,
        algorithm_name=method,
        target_domain=target_domain,
        verbose=verbose,
    )



def train_one_run(
    algorithm_name: str,
    target_domain: str,
    hparams: dict,
    seed: int = 42,
    data_dir: str = './data/nfv2/processed',
    max_epochs: int = 50,
    early_stop_patience: int = 10,
    early_stop_metric: str = 'val_auroc_macro',
    device: Optional[str] = None,
    verbose: bool = True,
    save_dir: Optional[str] = None,
    use_invalid_indicators: bool = True,
    target_few_shot_n: int = 0,
    target_few_shot_seed: int = 0,
    target_few_shot_balanced: bool = True,
    target_few_shot_policy: str = DEFAULT_FEW_SHOT_POLICY,
    target_only: bool = False,
    attack_family_policy: str = "exact",
) -> dict:
    """
     + .

    :
    1. :LODODataLoader.prepare()
    2. ()
    3. 
    4. : epoch  val -> early stopping based on early_stop_metric
    5.  best checkpoint :in-domain + cross-domain + val

    :
        dict,:
            - algorithm: str
            - target_domain: str
            - seed: int
            - hparams: dict
            - val_attack_f1_macro: float  (best )
            - val_attack_f1_per_domain: dict
            - in_domain: dict (evaluate_full )
            - cross_domain: dict (evaluate_full )
            - degradation: float  (in - cross)
            - best_epoch: int
            - train_time_sec: float
    """
    allowed_early_stop_metrics = {
        'val_attack_f1_macro',
        'val_auroc_macro',
        'val_auprc_macro',
        'val_tpr_at_fpr_1pct_macro',
        'val_unique_attack_f1_macro',
        'val_unique_auroc_macro',
    }
    if early_stop_metric not in allowed_early_stop_metrics:
        raise ValueError(
            f"Unsupported early_stop_metric={early_stop_metric!r}; "
            f"allowed={sorted(allowed_early_stop_metrics)}"
        )

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    t_start = time.time()
    seed_everything(seed)

    loader = LODODataLoader(
        data_dir=data_dir,
        target_domain=target_domain,
        use_invalid_indicators=use_invalid_indicators,
        target_few_shot_n=target_few_shot_n,
        target_few_shot_seed=target_few_shot_seed,
        target_few_shot_balanced=target_few_shot_balanced,
        target_few_shot_policy=target_few_shot_policy,
        attack_family_policy=attack_family_policy,
    )
    loader.prepare()

    use_attack_family = algorithm_name.lower() in ('multitask_erm', 'multitask_erm_shared4')
    runtime_hparams = dict(hparams)
    attack_family_info = loader.get_attack_family_info()
    if use_attack_family:
        if attack_family_info.get("n_classes", 0) <= 1:
            raise ValueError(
                f"attack-family metadata unavailable or degenerate for {algorithm_name}/{target_domain}"
            )
        family_train = loader.get_source_train_attack_family()
        runtime_hparams['n_attack_families'] = int(attack_family_info['n_classes'])
        runtime_hparams['attack_family_labels'] = list(attack_family_info['labels'])
        runtime_hparams['attack_family_class_weights'] = _multiclass_inverse_freq_weights(
            family_train,
            n_classes=attack_family_info['n_classes'],
        ).tolist()

    train_datasets = loader.get_train_datasets(include_attack_family=use_attack_family)
    val_per_domain = loader.get_source_val_per_domain()  # {dom: (X, y)}
    X_in, y_in = loader.get_in_domain_eval()
    X_cross, y_cross = loader.get_cross_domain_eval_excluding_few_shot()
    if target_only:
        if target_few_shot_n <= 0:
            raise ValueError(
                "target_only=True requires target_few_shot_n>0 "
                "(no source data is used; few-shot rows ARE the training set)."
            )
    if target_few_shot_n > 0:
        if needs_domain_balanced_batch(algorithm_name):
            raise NotImplementedError(
                f"target_few_shot_n>0 with domain-balanced algorithm "
                f"{algorithm_name!r} is not supported. Use ERM or "
                f"multitask_erm (random batches); domain-balanced "
                f"few-shot variants need oversampling or fine-tune "
                f"scheduling, which is not implemented here."
            )
        fs_domain_label = 0 if target_only else len(loader.source_domains)
        few_shot_ds = loader.get_few_shot_dataset(
            domain_label=fs_domain_label,
            include_attack_family=use_attack_family,
            subset="train",
        )
        if few_shot_ds is not None and len(few_shot_ds) > 0:
            fs_val_X, fs_val_y = loader.get_few_shot_validation()
            if target_only:
                train_datasets = [few_shot_ds]
                # Cap batch_size so we get >= 4 mini-batches per epoch even on
                # tiny target slices (e.g. N=40 -> bs=10). Default bs=512 would
                # give a single full-batch per epoch and stall optimization.
                n_train_fs = len(few_shot_ds)
                hp_bs = int(runtime_hparams.get('batch_size', 512))
                runtime_hparams['batch_size'] = max(8, min(hp_bs, n_train_fs // 4))
            else:
                train_datasets = list(train_datasets) + [few_shot_ds]
            # All adaptation strategies consume the same N-label budget: the
            # fixed support-fit subset trains the model and support-val selects
            # the epoch. Source validation is not used for target adaptation.
            val_per_domain = {
                f'{loader.target_domain}_support_val': (fs_val_X, fs_val_y)
            }
            if verbose:
                aud = loader.get_few_shot_audit()
                mode = "target-only" if target_only else "source+target"
                print(
                    f"  [few-shot/{mode}] injected target slice as domain "
                    f"{fs_domain_label}: "
                    f"n={aud['selected_indices_count']} "
                    f"pos={aud['n_pos']} neg={aud['n_neg']} "
                    f"fit={aud['n_fit']} val={aud['n_val']} "
                    f"fixed_eval={aud['n_eval_after_exclude']}"
                )
    val_raw_per_domain = loader.get_source_val_unscaled(per_domain=True)
    val_unique_weights_per_domain = {
        dom: unique_row_weights(X_raw)
        for dom, (X_raw, _) in val_raw_per_domain.items()
    }
    X_in_raw, _ = loader.get_in_domain_eval_unscaled()
    X_cross_raw, _ = loader.get_cross_domain_eval_unscaled_excluding_few_shot()
    in_unique_weights = unique_row_weights(X_in_raw)
    cross_unique_weights = unique_row_weights(X_cross_raw)

    batch_size = int(runtime_hparams.get('batch_size', 512))
    domain_balanced = needs_domain_balanced_batch(algorithm_name)
    n_train_domains = len(train_datasets)

    if domain_balanced and batch_size % n_train_domains != 0:
        batch_size = max(n_train_domains, (batch_size // n_train_domains) * n_train_domains)
        runtime_hparams['batch_size'] = batch_size
        if verbose:
            print(f"  [batch] adjusted batch_size={batch_size} for {n_train_domains} domains")

    if device == 'cuda' or (isinstance(device, str) and device.startswith('cuda')):
        from nidsbed.datasets import GPUResidentBatchIterator
        train_loader = GPUResidentBatchIterator(
            train_datasets,
            batch_size=batch_size,
            domain_balanced=domain_balanced,
            device=device,
            seed=seed,
        )
    else:
        train_loader = build_train_loader(
            train_datasets,
            batch_size=batch_size,
            domain_balanced=domain_balanced,
            num_workers=0 if sys.platform.startswith('win') else 2,
            seed=seed,
        )


    class_weights_per_domain = {}
    for i, ds in enumerate(train_datasets):
        y_np = ds.y.numpy()
        class_weights_per_domain[i] = get_class_weights(y_np, clip_max=20.0, device=device)

    AlgoCls = get_algorithm_class(algorithm_name)
    algorithm = AlgoCls(
        in_dim=loader.n_features,
        n_classes=2,
        hparams=runtime_hparams,
        class_weights_per_domain=class_weights_per_domain,
        device=device,
    )

    if verbose:
        n_params = sum(p.numel() for p in algorithm.parameters() if p.requires_grad)
        print(
            f"  algorithm={algorithm_name.upper()}, "
            f"trainable_params={n_params:,}, "
            f"batch_size={batch_size}, domain_balanced={domain_balanced}"
        )
        if use_attack_family:
            print(
                f"    attack_family_classes={attack_family_info['n_classes']}, "
                f"family_lambda={runtime_hparams.get('family_lambda', 'N/A')}"
            )

    if hasattr(algorithm, 'set_total_steps'):
        total_steps = max_epochs * len(train_loader)
        algorithm.set_total_steps(total_steps)

    best_val_score = -float('inf')
    best_epoch = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        algorithm.train()
        t_epoch = time.time()
        epoch_losses = {}

        for batch in train_loader:
            loss_dict = algorithm.update(batch)
            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, []).append(v)

        val_info = evaluate_source_val_macro_avg(
            algorithm,
            val_per_domain,
            device=device,
            unique_weights_per_domain=val_unique_weights_per_domain,
        )
        val_score = val_info.get(early_stop_metric)
        if val_score is None or not np.isfinite(float(val_score)):
            raise RuntimeError(
                f"Early-stop metric {early_stop_metric!r} is unavailable or NaN "
                f"for {algorithm_name}/{target_domain}/seed{seed} at epoch {epoch}."
            )
        val_score = float(val_score)

        avg_losses = {k: float(np.mean(v)) for k, v in epoch_losses.items()}

        if verbose and (epoch == 1 or epoch % 5 == 0):
            loss_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
            print(f"  Ep {epoch:02d}/{max_epochs} [{format_time(time.time()-t_epoch)}] "
                  f"{loss_str}, {early_stop_metric}={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in algorithm.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch={best_epoch}, best {early_stop_metric}={best_val_score:.4f})")
                break

    if best_state is not None:
        algorithm.load_state_dict(best_state)

    final_val_info = evaluate_source_val_macro_avg(
        algorithm,
        val_per_domain,
        device=device,
        unique_weights_per_domain=val_unique_weights_per_domain,
    )

    in_domain_metrics = evaluate_full(
        algorithm,
        X_in,
        y_in,
        device=device,
        unique_weights=in_unique_weights,
    )

    cross_domain_metrics = evaluate_full(
        algorithm,
        X_cross,
        y_cross,
        device=device,
        unique_weights=cross_unique_weights,
    )

    degradation = in_domain_metrics['attack_f1'] - cross_domain_metrics['attack_f1']
    train_time = time.time() - t_start

    result = {
        'algorithm': algorithm_name,
        'algorithm_metadata': get_algorithm_metadata(algorithm_name),
        'target_domain': target_domain,
        'seed': seed,
        'hparams': runtime_hparams,
        'early_stop_metric': early_stop_metric,
        'best_val_metric': best_val_score,
        'val_attack_f1_macro': final_val_info['val_attack_f1_macro'],
        'val_auroc_macro': final_val_info.get('val_auroc_macro'),
        'val_auprc_macro': final_val_info.get('val_auprc_macro'),
        'val_tpr_at_fpr_1pct_macro': final_val_info.get('val_tpr_at_fpr_1pct_macro'),
        'val_unique_attack_f1_macro': final_val_info.get('val_unique_attack_f1_macro'),
        'val_unique_auroc_macro': final_val_info.get('val_unique_auroc_macro'),
        'val_attack_f1_per_domain': final_val_info['val_attack_f1_per_domain'],
        'val_metrics_per_domain': final_val_info.get('val_metrics_per_domain'),
        'in_domain': in_domain_metrics,
        'cross_domain': cross_domain_metrics,
        'degradation': degradation,
        'best_epoch': best_epoch,
        'train_time_sec': train_time,
        'split_policy': loader.split_audit.get('policy'),
        'feature_preprocessing_policy': loader.preprocessing_audit.get('policy'),
        'feature_preprocessing_audit': loader.preprocessing_audit,
        'target_few_shot_audit': loader.get_few_shot_audit(),
        'target_few_shot_n': int(target_few_shot_n),
        'target_few_shot_seed': int(target_few_shot_seed),
        'target_few_shot_balanced': bool(target_few_shot_balanced),
        'target_few_shot_policy': target_few_shot_policy,
        'target_only': bool(target_only),
        'attack_family_info': attack_family_info if use_attack_family else None,
    }

    if verbose:
        print(f"  OK  [{format_time(train_time)}]: "
              f"val={final_val_info['val_attack_f1_macro']:.4f}, "
              f"in={in_domain_metrics['attack_f1']:.4f}, "
              f"cross={cross_domain_metrics['attack_f1']:.4f}, "
              f"degrade={degradation:.4f}")
    
    if save_dir is not None:
        import os
        from nidsbed.model_selection import predict_batched
        os.makedirs(save_dir, exist_ok=True)

        # (a) checkpoint
        if best_state is not None:
            ckpt_payload = {
                'algorithm': algorithm_name,
                'algorithm_metadata': get_algorithm_metadata(algorithm_name),
                'target_domain': target_domain,
                'seed': seed,
                'hparams': runtime_hparams,
                'early_stop_metric': early_stop_metric,
                'best_val_metric': best_val_score,
                'best_epoch': best_epoch,
                'state_dict': best_state,
                'in_dim': loader.n_features,
                'n_classes': 2,
                'attack_family_info': attack_family_info if use_attack_family else None,
                'target_few_shot_n': int(target_few_shot_n),
                'target_few_shot_seed': int(target_few_shot_seed),
                'target_few_shot_balanced': bool(target_few_shot_balanced),
                'target_few_shot_policy': target_few_shot_policy,
                'target_only': bool(target_only),
                'target_few_shot_audit': loader.get_few_shot_audit(),
            }
            torch.save(ckpt_payload, os.path.join(save_dir, 'checkpoint.pt'))

        calibration_meta = save_eval_artifacts(
            algorithm=algorithm,
            loader=loader,
            train_datasets=train_datasets,
            seed=seed,
            device=device,
            save_dir=save_dir,
            algorithm_name=algorithm_name,
            target_domain=target_domain,
            verbose=verbose,
        )
        result['calibration'] = calibration_meta

        if verbose:
            print(f"  [save] -> {save_dir}/")

        result['save_dir'] = save_dir

    return result


# ============================================================
# ============================================================

def _cli_main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Train one LODO DG or few-shot DA run and optionally save JSON."
    )
    parser.add_argument(
        '--algorithm', required=True,
        help='DG method: erm, irm, coral, groupdro, dann, fishr, boda, multitask_erm',
    )
    parser.add_argument('--target', required=True, choices=['unsw', 'cic', 'bot', 'ton'])
    parser.add_argument('--data_dir', default='./DATA/nfv2_processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--early_stop_metric', default='val_auroc_macro',
                        choices=[
                            'val_attack_f1_macro',
                            'val_auroc_macro',
                            'val_auprc_macro',
                            'val_tpr_at_fpr_1pct_macro',
                            'val_unique_attack_f1_macro',
                            'val_unique_auroc_macro',
                        ])
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default=None, help='Optional JSON path for run metrics')
    parser.add_argument('--save_dir', default=None,
                        help='Optional directory for checkpoint.pt and score arrays')


    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--feat_dim', type=int, default=64)
    parser.add_argument('--family_lambda', type=float, default=None)
    parser.add_argument('--tail_lambda', type=float, default=None)
    parser.add_argument('--tail_frac', type=float, default=None)
    parser.add_argument('--tail_margin', type=float, default=None)
    parser.add_argument('--tail_abs_lambda', type=float, default=None)
    parser.add_argument('--tail_abs_threshold', type=float, default=None)

    parser.add_argument('--target_few_shot_n', type=int, default=0,
                        help='If >0, switch to few-shot DA: include this many '
                             'class-stratified labelled target rows as a 4th '
                             'training domain. Strict-DG default is 0.')
    parser.add_argument('--target_few_shot_seed', type=int, default=0,
                        help='Seed for the frozen few-shot target subset.')
    parser.add_argument('--target_few_shot_unbalanced', action='store_true',
                        help='Sample target subset uniformly at random instead '
                             'of class-stratified (default is balanced).')
    parser.add_argument(
        '--target_few_shot_policy',
        default=DEFAULT_FEW_SHOT_POLICY,
        choices=[FEW_SHOT_POLICY_GROUP_NESTED, FEW_SHOT_POLICY_LEGACY],
        help='Support/evaluation protocol. The default samples unique target-train '
             'groups and evaluates on the fixed target-test split.',
    )
    parser.add_argument('--target_only', action='store_true',
                        help='Train ONLY on the few-shot target rows '
                             '(no source data). Requires --target_few_shot_n>0.')

    args = parser.parse_args()

    hparams = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'hidden': args.hidden,
        'feat_dim': args.feat_dim,
    }
    optional_hparams = {
        'family_lambda': args.family_lambda,
        'tail_lambda': args.tail_lambda,
        'tail_frac': args.tail_frac,
        'tail_margin': args.tail_margin,
        'tail_abs_lambda': args.tail_abs_lambda,
        'tail_abs_threshold': args.tail_abs_threshold,
    }
    for key, value in optional_hparams.items():
        if value is not None:
            hparams[key] = value

    result = train_one_run(
        algorithm_name=args.algorithm,
        target_domain=args.target,
        hparams=hparams,
        seed=args.seed,
        data_dir=args.data_dir,
        max_epochs=args.max_epochs,
        early_stop_patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        device=args.device,
        verbose=True,
        save_dir=args.save_dir,
        target_few_shot_n=args.target_few_shot_n,
        target_few_shot_seed=args.target_few_shot_seed,
        target_few_shot_balanced=not args.target_few_shot_unbalanced,
        target_few_shot_policy=args.target_few_shot_policy,
        target_only=args.target_only,
    )

    if args.output:
        from nidsbed.lib.misc import save_json
        save_json(result, args.output)
        print(f"\nWrote results to {args.output}")


if __name__ == '__main__':
    _cli_main()
