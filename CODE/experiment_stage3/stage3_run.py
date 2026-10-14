"""
Stage-3 driver: hyperparameter search and final-seed runs per (method, target).

Modes:
    search : sample ``n_search`` HP configs over search seeds (default 42, 123)
    final  : pick the best HP by source-validation ``selection_metric``, then
             run final seeds (default 2026, 2027, 2028) and write summaries
    all    : run search then final

Output layout:
    results/stage3/
        {method}_{target}/
            hp_grid.json
            meta.json
            search/hp{0..7}.json
            final/best_hp.json, seed{2026,2027,2028}.json, summary.json

Resume behaviour:
    - skip cells whose JSON already has ``status=success``
    - delete and retry corrupt JSON files
    - write via ``.tmp`` + ``os.replace`` for atomic updates

Examples:
    python CODE/run_paper.py stage3 --method fishr --target unsw --mode all
    python CODE/run_paper.py stage3 --method fishr --target unsw --mode all --dry_run
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))

import argparse
import gc
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
COMMON_ROOT = PROJECT_ROOT / "CODE" / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

import numpy as np
import torch

from CODE.experiment_stage3.train import backfill_calibration_artifacts, train_one_run
from nidsbed.algorithm_metadata import get_algorithm_metadata
from nidsbed.datasets import FEATURE_PREPROCESS_POLICY, SPLIT_POLICY
from nidsbed.lib.hp_space import (
    build_full_grid,
    sample_hp_configs,
    grid_info,
    hp_rng_seed,
)


# =========================================================================
# =========================================================================
DEFAULT_N_SEARCH = 8
DEFAULT_SEARCH_SEEDS = [42, 123]
DEFAULT_FINAL_SEEDS = [2026, 2027, 2028]
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 10
DEFAULT_OUTPUT_ROOT = 'RESULTS/stage3_lodo/stage3'
DEFAULT_SELECTION_METRIC = 'val_auroc_macro'

ALL_METHODS = ['erm', 'multitask_erm', 'irm', 'coral', 'deepcoral', 'groupdro',
               'dann', 'fishr', 'boda']
ALL_TARGETS = ['unsw', 'cic', 'bot', 'ton']


# =========================================================================
# =========================================================================
def collect_env_meta():
    """Collect runtime metadata only. No version-control dependency is used."""
    gpu_name = None
    cuda_version = None
    cudnn_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        try:
            cudnn_version = str(torch.backends.cudnn.version())
        except Exception:
            cudnn_version = None

    return {
        'python_version':   platform.python_version(),
        'torch_version':    torch.__version__,
        'numpy_version':    np.__version__,
        'cuda_version':     cuda_version,
        'cudnn_version':    cudnn_version,
        'gpu_name':         gpu_name,
        'hostname':         socket.gethostname(),
        'platform':         platform.platform(),
    }


# =========================================================================
# =========================================================================
def atomic_write_json(path: Path, data: dict):
    """Atomically write JSON via a ``.tmp`` file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_load_json(path: Path):
    """Load JSON from disk; return None and delete corrupt files on failure."""
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] corrupt JSON at {path} ({e}); deleting file.")
        try:
            path.unlink()
        except OSError:
            pass
        return None


BASE_FINAL_ARTIFACTS = (
    'checkpoint.pt',
    'target_scores.npy',
    'target_labels.npy',
    'target_unique_weights.npy',
    'source_scores.npy',
    'source_labels.npy',
    'source_X_sample.npy',
)

CALIBRATION_FINAL_ARTIFACTS = (
    'target_logits.npy',
    'target_scores_calibrated.npy',
    'source_val_logits.npy',
    'source_val_scores.npy',
    'source_val_scores_calibrated.npy',
    'source_val_labels.npy',
    'source_val_unique_weights.npy',
    'source_intest_logits.npy',
    'source_intest_scores.npy',
    'source_intest_labels.npy',
    'source_intest_unique_weights.npy',
    'calibration.json',
)

FINAL_ARTIFACTS = BASE_FINAL_ARTIFACTS + CALIBRATION_FINAL_ARTIFACTS


def final_artifacts_status(final_dir: Path, seed: int,
                           include_calibration: bool = True) -> tuple[bool, list[str]]:
    """Return whether a successful final seed also has all downstream artifacts."""
    art_dir = final_dir / f"seed{seed}_artifacts"
    names = FINAL_ARTIFACTS if include_calibration else BASE_FINAL_ARTIFACTS
    missing = []
    for name in names:
        path = art_dir / name
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(str(path.relative_to(final_dir)))
    return len(missing) == 0, missing


# =========================================================================
# =========================================================================
def reset_global_state(seed: int):
    """Reset RNG state and CUDA cache before each isolated training run."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# =========================================================================
# =========================================================================
def run_single(method: str, target: str, hparams: dict, seed: int,
               data_dir: str, max_epochs: int, patience: int,
               early_stop_metric: str, device: str, verbose: bool,
               save_dir: str = None) -> dict:
    """Run one ``train_one_run`` call and return ``{ok: True, result}`` or error info."""
    reset_global_state(seed)
    try:
        result = train_one_run(
            algorithm_name=method,
            target_domain=target,
            hparams=hparams,
            seed=seed,
            data_dir=data_dir,
            max_epochs=max_epochs,
            early_stop_patience=patience,
            early_stop_metric=early_stop_metric,
            device=device,
            verbose=verbose,
            save_dir=save_dir,
        )
        return {'ok': True, 'result': result}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [ERROR] training run failed: {type(e).__name__}: {e}")
        print(tb)
        return {
            'ok': False,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': tb,
        }


def make_run_record(method: str, target: str, mode: str, hp_id,
                    seed: int, hparams: dict, run_result: dict,
                    env_meta: dict) -> dict:
    """Build the per-run JSON record written to disk for one HP / seed."""
    record = {
        'method':     method,
        'algorithm_metadata': get_algorithm_metadata(method),
        'target':     target,
        'mode':       mode,
        'hp_id':      hp_id,
        'seed':       seed,
        'hparams':    hparams,
        'timestamp':  datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        **env_meta,
    }
    if run_result['ok']:
        r = run_result['result']
        record.update({
            'status':                    'success',
            'in_attack_f1':              r['in_domain']['attack_f1'],
            'cross_attack_f1':           r['cross_domain']['attack_f1'],
            'in_auroc':                  r['in_domain'].get('auroc'),
            'cross_auroc':               r['cross_domain'].get('auroc'),
            'in_auprc':                  r['in_domain'].get('auprc'),
            'cross_auprc':               r['cross_domain'].get('auprc'),
            'in_tpr_at_fpr_1pct':        r['in_domain'].get('tpr_at_fpr_1pct'),
            'cross_tpr_at_fpr_1pct':     r['cross_domain'].get('tpr_at_fpr_1pct'),
            'cross_unique_attack_f1':    r['cross_domain'].get('unique_attack_f1'),
            'cross_unique_auroc':        r['cross_domain'].get('unique_auroc'),
            'cross_unique_auprc':        r['cross_domain'].get('unique_auprc'),
            'in_precision':              r['in_domain'].get('precision'),
            'in_recall':                 r['in_domain'].get('recall'),
            'cross_precision':           r['cross_domain'].get('precision'),
            'cross_recall':              r['cross_domain'].get('recall'),
            'val_attack_f1_macro':       r['val_attack_f1_macro'],
            'val_auroc_macro':           r.get('val_auroc_macro'),
            'val_auprc_macro':           r.get('val_auprc_macro'),
            'val_tpr_at_fpr_1pct_macro': r.get('val_tpr_at_fpr_1pct_macro'),
            'val_unique_attack_f1_macro': r.get('val_unique_attack_f1_macro'),
            'val_unique_auroc_macro':    r.get('val_unique_auroc_macro'),
            'val_attack_f1_per_domain':  r['val_attack_f1_per_domain'],
            'val_metrics_per_domain':    r.get('val_metrics_per_domain'),
            'degradation':               r['degradation'],
            'best_epoch':                r['best_epoch'],
            'early_stop_metric':         r.get('early_stop_metric'),
            'best_val_metric':           r.get('best_val_metric'),
            'train_time_sec':            r['train_time_sec'],
            'split_policy':              r.get('split_policy'),
            'feature_preprocessing_policy': r.get('feature_preprocessing_policy'),
        })
    else:
        record.update({
            'status':         'error',
            'error_type':     run_result['error_type'],
            'error_message': run_result['error_message'],
            'traceback':      run_result['traceback'],
        })
    return record


# =========================================================================
# =========================================================================
def _aggregate_search_seed_records(method: str, target: str, hp_id: int, hparams: dict,
                                   records: list[dict], selection_metric: str,
                                   env_meta: dict) -> dict:
    ok_records = [r for r in records if r.get('status') == 'success']
    if not ok_records:
        return {
            'method': method,
            'algorithm_metadata': get_algorithm_metadata(method),
            'target': target,
            'mode': 'search',
            'hp_id': hp_id,
            'seed': 'search_mean',
            'hparams': hparams,
            'status': 'error',
            'selection_metric': selection_metric,
            'error_message': 'no successful search seeds',
            **env_meta,
        }

    def _mean(key):
        vals = [r.get(key) for r in ok_records]
        vals = [float(v) for v in vals if v is not None and not np.isnan(float(v))]
        return float(np.mean(vals)) if vals else None

    out = {
        'method': method,
        'algorithm_metadata': get_algorithm_metadata(method),
        'target': target,
        'mode': 'search',
        'hp_id': hp_id,
        'seed': 'search_mean',
        'search_seeds': [int(r['seed']) for r in ok_records],
        'n_search_seeds_success': len(ok_records),
        'early_stop_metric': ok_records[0].get('early_stop_metric'),
        'hparams': hparams,
        'status': 'success',
        'selection_metric': selection_metric,
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        **env_meta,
    }
    for key in [
        'val_attack_f1_macro',
        'val_auroc_macro',
        'val_auprc_macro',
        'val_tpr_at_fpr_1pct_macro',
        'val_unique_attack_f1_macro',
        'val_unique_auroc_macro',
        'in_attack_f1',
        'cross_attack_f1',
        'in_auroc',
        'cross_auroc',
    ]:
        out[key] = _mean(key)
    if out.get(selection_metric) is None:
        raise RuntimeError(
            f"Selection metric {selection_metric!r} is unavailable for {method}/{target}/hp{hp_id}"
        )
    out['per_search_seed'] = [
        {
            'seed': r.get('seed'),
            'status': r.get('status'),
            selection_metric: r.get(selection_metric),
            'val_attack_f1_macro': r.get('val_attack_f1_macro'),
            'val_auroc_macro': r.get('val_auroc_macro'),
            'val_auprc_macro': r.get('val_auprc_macro'),
        }
        for r in records
    ]
    return out


def run_search(method, target, n_search, search_seeds, data_dir, output_dir, max_epochs,
               patience, early_stop_metric, device, verbose, dry_run, env_meta,
               selection_metric):
    """Run HP search over n configs with resume support."""
    search_dir = output_dir / 'search'
    search_dir.mkdir(parents=True, exist_ok=True)

    info = grid_info(method, target, n_search)
    print(f"  [hp_grid] sampling_mode={info['sampling_mode']}, "
          f"full_grid={info['full_grid_size']}, "
          f"n_sampled={info['n_sampled']}, "
          f"rng_seed_hex={info['rng_seed_hex']}")

    grid_path = output_dir / 'hp_grid.json'
    if grid_path.exists():
        existing_grid = safe_load_json(grid_path)
        if existing_grid is not None:
            existing_sig = (existing_grid.get('full_grid_size'),
                            existing_grid.get('n_sampled'),
                            existing_grid.get('rng_seed_hex'),
                            existing_grid.get('sampling_mode'))
            current_sig = (info['full_grid_size'], info['n_sampled'],
                           info['rng_seed_hex'], info['sampling_mode'])
            if existing_sig != current_sig:
                print(f"  [WARN] hp_grid.json signature mismatch:")
                print(f"    existing: full_grid={existing_sig[0]}, n_sampled={existing_sig[1]}, "
                      f"seed={existing_sig[2]}, mode={existing_sig[3]}")
                print(f"    current:  full_grid={current_sig[0]}, n_sampled={current_sig[1]}, "
                      f"seed={current_sig[2]}, mode={current_sig[3]}")
                print("  Existing search JSON files do not match the current HP grid.")
                print(f"  Start a fresh run or remove the stale search directory: {output_dir}")
                sys.exit(4)
    elif not dry_run:
        atomic_write_json(grid_path, info)

    configs = sample_hp_configs(method, target, n_search)

    n_done = 0
    n_skip = 0
    n_error_prev = 0
    for hp_id, hparams in configs:
        out_path = search_dir / f"hp{hp_id}.json"
        existing = safe_load_json(out_path)
        if existing is not None and existing.get('status') == 'success':
            if existing.get('selection_metric') == selection_metric and existing.get(selection_metric) is not None:
                n_skip += 1
                if verbose:
                    print(f"  [SKIP] hp{hp_id} ({selection_metric}={existing.get(selection_metric, 'N/A')})")
                continue
            print(f"  [REPAIR] hp{hp_id} aggregate missing selection metric {selection_metric}")
        if existing is not None and existing.get('status') == 'error':
            n_error_prev += 1
            print(f"  [RETRY] hp{hp_id} previous error "
                  f"({existing.get('error_type', '?')})")

        print(f"\n>>> search run [{method}/{target}] hp{hp_id} / {len(configs)-1}")
        if verbose:
            diff_keys = set(hparams.keys()) - {'batch_size', 'hidden', 'feat_dim',
                                               'disc_hidden', 'boda_warmup_iters',
                                               'boda_min_samples', 'mix_p', 'mix_alpha'}
            print(f"    hparams: { {k: hparams[k] for k in sorted(diff_keys)} }")

        if dry_run:
            print(f"    [DRY RUN] skipped")
            continue

        seed_records = []
        for search_seed in search_seeds:
            seed_path = search_dir / f"hp{hp_id}_seed{search_seed}.json"
            seed_existing = safe_load_json(seed_path)
            if (
                seed_existing is not None
                and seed_existing.get('status') == 'success'
                and seed_existing.get(selection_metric) is not None
            ):
                seed_records.append(seed_existing)
                if verbose:
                    print(f"    [SKIP] hp{hp_id} seed{search_seed}")
                continue
            run_result = run_single(
                method, target, hparams, seed=search_seed,
                data_dir=data_dir, max_epochs=max_epochs, patience=patience,
                early_stop_metric=early_stop_metric, device=device, verbose=verbose,
            )
            record = make_run_record(method, target, 'search', hp_id, search_seed,
                                     hparams, run_result, env_meta)
            atomic_write_json(seed_path, record)
            seed_records.append(record)
        aggregate = _aggregate_search_seed_records(
            method, target, hp_id, hparams, seed_records, selection_metric, env_meta
        )
        atomic_write_json(out_path, aggregate)
        n_done += 1

    print(f"\n[search] {method}/{target}: "
          f"{n_done} ran, {n_error_prev} prior errors, "
          f"{n_skip} skipped, {len(configs)} HP configs total")


def select_best_hp(output_dir, selection_metric: str):
    """Select HP by the configured source-validation metric. Ties use hp_id."""
    search_dir = output_dir / 'search'
    all_runs = []
    for json_path in sorted(search_dir.glob('hp*.json')):
        if re.match(r"hp\d+_seed\d+\.json$", json_path.name):
            continue
        data = safe_load_json(json_path)
        if data is None or data.get('status') != 'success':
            continue
        if data.get(selection_metric) is None:
            continue
        all_runs.append(data)

    if not all_runs:
        raise RuntimeError(f"No successful search runs in {search_dir}. "
                           f"Run --mode search first.")

    all_runs.sort(key=lambda r: (-float(r[selection_metric]), r['hp_id']))
    return all_runs[0]


# =========================================================================
# =========================================================================
def run_final(method, target, seeds, data_dir, output_dir, max_epochs,
              patience, early_stop_metric, device, verbose, dry_run, env_meta,
              selection_metric):
    """Run final seeds for the best HP selected on source validation."""
    final_dir = output_dir / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)

    best_hp_path = final_dir / 'best_hp.json'
    best_record = safe_load_json(best_hp_path) if best_hp_path.exists() else None
    if best_record is not None:
        if best_record.get('selection_metric') != selection_metric:
            raise RuntimeError(
                f"Existing best_hp.json was selected by {best_record.get('selection_metric')!r}, "
                f"current selection_metric is {selection_metric!r}. Start a fresh run."
            )
        if best_record.get('early_stop_metric') not in (None, early_stop_metric):
            raise RuntimeError(
                f"Existing best_hp.json used early_stop_metric={best_record.get('early_stop_metric')!r}, "
                f"current early_stop_metric is {early_stop_metric!r}. Start a fresh run."
            )
        repaired_best_record = False
        if best_record.get('early_stop_metric') is None:
            best_record['early_stop_metric'] = early_stop_metric
            repaired_best_record = True
        if best_record.get('search_seeds') is None or best_record.get('n_search_seeds_success') is None:
            try:
                best_run = select_best_hp(output_dir, selection_metric)
                best_record.setdefault('selection_metric_value', best_run.get(selection_metric))
                best_record['search_seeds'] = best_run.get('search_seeds')
                best_record['n_search_seeds_success'] = best_run.get('n_search_seeds_success')
                repaired_best_record = True
            except Exception as e:
                print(f"  [WARN] could not repair best_hp search seed metadata: {e}")
        if repaired_best_record and not dry_run:
            atomic_write_json(best_hp_path, best_record)
        print(f"  [final] reusing best HP (hp_id={best_record['hp_id']})")
    elif dry_run:
        configs = sample_hp_configs(method, target, DEFAULT_N_SEARCH)
        stub_hp_id, stub_hparams = configs[0]
        best_record = {
            'hp_id':                stub_hp_id,
            'hparams':               stub_hparams,
            'selection_metric':       selection_metric,
            'early_stop_metric':      early_stop_metric,
            'val_attack_f1_macro':   float('nan'),
            selection_metric:         float('nan'),
            'in_attack_f1_search':   float('nan'),
        }
        print(f"  [final] [DRY RUN] using stub hp{stub_hp_id} (search not run)")
    else:
        best_run = select_best_hp(output_dir, selection_metric)
        best_record = {
            'hp_id':                best_run['hp_id'],
            'hparams':              best_run['hparams'],
            'selection_metric':      selection_metric,
            'early_stop_metric':     early_stop_metric,
            'selection_metric_value': best_run.get(selection_metric),
            'search_seeds':          best_run.get('search_seeds'),
            'n_search_seeds_success': best_run.get('n_search_seeds_success'),
            'val_attack_f1_macro':  best_run['val_attack_f1_macro'],
            'val_auroc_macro':      best_run.get('val_auroc_macro'),
            'val_auprc_macro':      best_run.get('val_auprc_macro'),
            'val_tpr_at_fpr_1pct_macro': best_run.get('val_tpr_at_fpr_1pct_macro'),
            'in_attack_f1_search':  best_run['in_attack_f1'],
        }
        atomic_write_json(best_hp_path, best_record)
        metric_value = best_record.get('selection_metric_value')
        print(f"  [final] selected hp{best_record['hp_id']} "
              f"({selection_metric}={float(metric_value):.4f})")

    best_hparams = best_record['hparams']


    n_done = 0
    n_skip = 0
    for seed in seeds:
        out_path = final_dir / f"seed{seed}.json"
        existing = safe_load_json(out_path)
        base_artifacts_ok, base_missing_artifacts = final_artifacts_status(
            final_dir, seed, include_calibration=False
        )
        artifacts_ok, missing_artifacts = final_artifacts_status(
            final_dir, seed, include_calibration=True
        )
        if (
            existing is not None
            and existing.get('status') == 'success'
            and base_artifacts_ok
            and not artifacts_ok
            and not dry_run
        ):
            seed_artifacts_dir = str(final_dir / f"seed{seed}_artifacts")
            print(f"  [REPAIR] seed{seed} base artifacts exist; rebuilding calibration artifacts")
            try:
                backfill_calibration_artifacts(
                    method=method,
                    target_domain=target,
                    seed=seed,
                    data_dir=data_dir,
                    device=device,
                    save_dir=seed_artifacts_dir,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  [WARN] calibration backfill failed for seed{seed}: "
                      f"{type(e).__name__}: {e}")
            artifacts_ok, missing_artifacts = final_artifacts_status(
                final_dir, seed, include_calibration=True
            )

        if existing is not None and existing.get('status') == 'success' and artifacts_ok:
            n_skip += 1
            if verbose:
                print(f"  [SKIP] seed{seed} already complete "
                      f"(in_f1={existing.get('in_attack_f1', 'N/A')})")
            continue
        if existing is not None and existing.get('status') == 'success':
            if not base_artifacts_ok:
                print(f"  [REPAIR] seed{seed} JSON is success but base artifacts are incomplete: "
                      f"{', '.join(base_missing_artifacts)}")
            elif not artifacts_ok:
                print(f"  [REPAIR] seed{seed} JSON is success but calibration artifacts are incomplete: "
                      f"{', '.join(missing_artifacts)}")

        print(f"\n>>> final run [{method}/{target}] seed={seed} (hp{best_record['hp_id']})")
        if dry_run:
            print(f"    [DRY RUN] skipped")
            continue

        seed_artifacts_dir = str(final_dir / f"seed{seed}_artifacts")
        run_result = run_single(
            method, target, best_hparams, seed=seed,
            data_dir=data_dir, max_epochs=max_epochs, patience=patience,
            early_stop_metric=early_stop_metric, device=device, verbose=verbose,
            save_dir=seed_artifacts_dir,
        )

        record = make_run_record(method, target, 'final', best_record['hp_id'], seed,
                                 best_hparams, run_result, env_meta)
        atomic_write_json(out_path, record)
        n_done += 1

    summary_path = final_dir / 'summary.json'
    completed = []
    for seed in seeds:
        rec = safe_load_json(final_dir / f"seed{seed}.json")
        artifacts_ok, missing_artifacts = final_artifacts_status(final_dir, seed)
        if rec is not None and rec.get('status') == 'success' and artifacts_ok:
            completed.append(rec)
        elif rec is not None and rec.get('status') == 'success':
            print(f"  [INCOMPLETE] seed{seed} success JSON but missing artifacts: "
                  f"{', '.join(missing_artifacts)}")

    if len(completed) == len(seeds) and not dry_run:
        def safe_std(arr):
            return float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        def agg(key):
            """ completed  key, None/NaN, (mean, std).
            None/ NaN  (None, None), summary."""
            vals = [r.get(key) for r in completed]
            arr = np.array([v for v in vals if v is not None], dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return None, None
            return float(arr.mean()), safe_std(arr)

        in_f1_m,    in_f1_s    = agg('in_attack_f1')
        cross_f1_m, cross_f1_s = agg('cross_attack_f1')
        in_au_m,    in_au_s    = agg('in_auroc')
        cross_au_m, cross_au_s = agg('cross_auroc')
        in_pr_m,    in_pr_s    = agg('in_precision')
        in_rc_m,    in_rc_s    = agg('in_recall')
        cross_pr_m, cross_pr_s = agg('cross_precision')
        cross_rc_m, cross_rc_s = agg('cross_recall')
        in_ap_m,    in_ap_s    = agg('in_auprc')
        cross_ap_m, cross_ap_s = agg('cross_auprc')
        in_tpr1_m,  in_tpr1_s  = agg('in_tpr_at_fpr_1pct')
        cross_tpr1_m, cross_tpr1_s = agg('cross_tpr_at_fpr_1pct')
        cross_unique_f1_m, cross_unique_f1_s = agg('cross_unique_attack_f1')
        cross_unique_au_m, cross_unique_au_s = agg('cross_unique_auroc')
        cross_unique_ap_m, cross_unique_ap_s = agg('cross_unique_auprc')
        deg_m,      deg_s      = agg('degradation')

        summary = {
            'method':                method,
            'algorithm_metadata':    get_algorithm_metadata(method),
            'target':                target,
            'hp_id':                 best_record['hp_id'],
            'hparams':               best_hparams,
            'selection_metric':       selection_metric,
            'early_stop_metric':      early_stop_metric,
            'search_seeds':           best_record.get('search_seeds'),
            'n_search_seeds_success': best_record.get('n_search_seeds_success'),
            'search_selection_metric_value': best_record.get('selection_metric_value'),
            'required_final_artifacts': list(FINAL_ARTIFACTS),
            'n_seeds':               len(seeds),
            'seeds':                 seeds,
            'in_attack_f1_mean':     in_f1_m,
            'in_attack_f1_std':      in_f1_s,
            'cross_attack_f1_mean':  cross_f1_m,
            'cross_attack_f1_std':   cross_f1_s,
            'in_auroc_mean':         in_au_m,
            'in_auroc_std':          in_au_s,
            'cross_auroc_mean':      cross_au_m,
            'cross_auroc_std':       cross_au_s,
            'in_auprc_mean':         in_ap_m,
            'in_auprc_std':          in_ap_s,
            'cross_auprc_mean':      cross_ap_m,
            'cross_auprc_std':       cross_ap_s,
            'in_tpr_at_fpr_1pct_mean': in_tpr1_m,
            'in_tpr_at_fpr_1pct_std': in_tpr1_s,
            'cross_tpr_at_fpr_1pct_mean': cross_tpr1_m,
            'cross_tpr_at_fpr_1pct_std': cross_tpr1_s,
            'cross_unique_attack_f1_mean': cross_unique_f1_m,
            'cross_unique_attack_f1_std': cross_unique_f1_s,
            'cross_unique_auroc_mean': cross_unique_au_m,
            'cross_unique_auroc_std': cross_unique_au_s,
            'cross_unique_auprc_mean': cross_unique_ap_m,
            'cross_unique_auprc_std': cross_unique_ap_s,
            'in_precision_mean':     in_pr_m,
            'in_precision_std':      in_pr_s,
            'in_recall_mean':        in_rc_m,
            'in_recall_std':         in_rc_s,
            'cross_precision_mean':  cross_pr_m,
            'cross_precision_std':   cross_pr_s,
            'cross_recall_mean':     cross_rc_m,
            'cross_recall_std':      cross_rc_s,
            # degradation
            'degradation_mean':      deg_m,
            'degradation_std':       deg_s,
            'per_seed':              [
                {'seed':              r['seed'],
                 'in_attack_f1':     r['in_attack_f1'],
                 'cross_attack_f1':  r['cross_attack_f1'],
                 'in_auroc':         r.get('in_auroc'),
                 'cross_auroc':      r.get('cross_auroc'),
                 'cross_auprc':      r.get('cross_auprc')}
                for r in completed
            ],
            'timestamp':             datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
        atomic_write_json(summary_path, summary)
        print(f"\n[final] in_f1 = {in_f1_m:.4f} +/- {in_f1_s:.4f}, "
              f"cross_f1 = {cross_f1_m:.4f} +/- {cross_f1_s:.4f}")
        if cross_au_m is not None:
            print(f"             in_auroc = {in_au_m:.4f} +/- {in_au_s:.4f}, "
                  f"cross_auroc = {cross_au_m:.4f} +/- {cross_au_s:.4f}")
    else:
        print(f"\n[final] {len(completed)}/{len(seeds)} seeds complete; "
              f"summary not written")
        if not dry_run:
            if summary_path.exists():
                summary_path.unlink()
            raise RuntimeError(
                f"Final runs incomplete for {method}/{target}: "
                f"{len(completed)}/{len(seeds)} seeds have complete JSON + artifacts."
            )

    print(f"[final] {method}/{target}: {n_done} ran, {n_skip} skipped")


# =========================================================================
# =========================================================================
def write_meta(output_dir, method, target, env_meta, n_search, search_seeds, seeds,
               max_epochs, patience, mode, dry_run, selection_metric,
               early_stop_metric):
    """Write resumable run metadata. Existing metadata is appended, not overwritten."""
    meta_path = output_dir / 'meta.json'
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    existing = safe_load_json(meta_path) if meta_path.exists() else None

    launch_record = {
        'timestamp':   now,
        'mode':        mode,
        'dry_run':     dry_run,
        'gpu_name':    env_meta['gpu_name'],
        'hostname':    env_meta['hostname'],
    }

    expected = {
        'method':           method,
        'algorithm_metadata': get_algorithm_metadata(method),
        'target':           target,
        'n_search':         n_search,
        'search_seeds':     search_seeds,
        'final_seeds':      seeds,
        'max_epochs':       max_epochs,
        'patience':         patience,
        'selection_metric':  selection_metric,
        'early_stop_metric': early_stop_metric,
        'split_policy':      SPLIT_POLICY,
        'feature_preprocessing_policy': FEATURE_PREPROCESS_POLICY,
        'hp_rng_seed_hex':  f"{hp_rng_seed(method, target):08x}",
    }

    if existing is None:
        meta = {
            **expected,
            'first_launch':     now,
            'first_env':        env_meta,
            'launches':         [launch_record],
        }
    else:
        mismatches = []
        for key, expected_value in expected.items():
            if key not in existing or existing.get(key) != expected_value:
                mismatches.append((key, existing.get(key), expected_value))
        if mismatches:
            print("[CONFIG MISMATCH] Existing Stage-3 outputs were produced with different settings:")
            for key, old, new in mismatches:
                print(f"  - {key}: existing={old!r}, current={new!r}")
            print(f"Refusing to resume into {output_dir}. Use the original settings or start a fresh run.")
            sys.exit(5)

        meta = existing
        for key, value in expected.items():
            meta.setdefault(key, value)
        launches = meta.get('launches', [])
        launches.append(launch_record)
        meta['launches'] = launches
        meta['last_launch'] = now
        meta['last_env'] = env_meta

    if not dry_run:
        atomic_write_json(meta_path, meta)


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deterministic', action='store_true',
                        help=' PyTorch deterministic mode, GPU ')
    parser.add_argument('--method', required=True, choices=ALL_METHODS)
    parser.add_argument('--target', required=True, choices=ALL_TARGETS)
    parser.add_argument('--mode', required=True, choices=['search', 'final', 'all'])
    parser.add_argument('--n_search', type=int, default=DEFAULT_N_SEARCH)
    parser.add_argument('--search_seeds', type=str,
                        default=','.join(str(s) for s in DEFAULT_SEARCH_SEEDS))
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in DEFAULT_FINAL_SEEDS))
    parser.add_argument('--selection_metric', default=DEFAULT_SELECTION_METRIC,
                        choices=[
                            'val_attack_f1_macro',
                            'val_auroc_macro',
                            'val_auprc_macro',
                            'val_tpr_at_fpr_1pct_macro',
                            'val_unique_attack_f1_macro',
                            'val_unique_auroc_macro',
                        ])
    parser.add_argument('--early_stop_metric', default='default',
                        choices=[
                            'default',
                            'val_attack_f1_macro',
                            'val_auroc_macro',
                            'val_auprc_macro',
                            'val_tpr_at_fpr_1pct_macro',
                            'val_unique_attack_f1_macro',
                            'val_unique_auroc_macro',
                        ],
                        help='Metric used to select the best epoch. default = selection_metric.')
    parser.add_argument('--data_dir', default='./DATA/nfv2_processed')
    parser.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--device', default=None)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.add_argument('--dry_run', action='store_true',
                        help=', run ')
    args = parser.parse_args()
    if args.deterministic:
        import os as _os
        _os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[deterministic] PyTorch deterministic mode ")

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    search_seeds = [int(s.strip()) for s in args.search_seeds.split(',') if s.strip()]
    if args.early_stop_metric == 'default':
        args.early_stop_metric = args.selection_metric

    output_dir = Path(args.output_root) / f"{args.method}_{args.target}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_meta = collect_env_meta()

    print("=" * 70)
    print(f"Stage 3 Run - method={args.method}  target={args.target}  mode={args.mode}")
    print(f"  output_dir = {output_dir}")
    print(f"  gpu = {env_meta['gpu_name']}")
    print(f"  n_search={args.n_search}  seeds={seeds}  "
          f"search_seeds={search_seeds}  selection_metric={args.selection_metric}  "
          f"early_stop_metric={args.early_stop_metric}  "
          f"max_epochs={args.max_epochs}  patience={args.patience}")
    if args.dry_run:
        print(f"  *** DRY RUN *** ()")
    print("=" * 70)

    write_meta(output_dir, args.method, args.target, env_meta,
               args.n_search, search_seeds, seeds, args.max_epochs, args.patience,
               args.mode, args.dry_run, args.selection_metric, args.early_stop_metric)

    t_start = time.time()

    if args.mode in ('search', 'all'):
        run_search(args.method, args.target, args.n_search, search_seeds, args.data_dir,
                   output_dir, args.max_epochs, args.patience, args.early_stop_metric,
                   args.device, args.verbose, args.dry_run, env_meta,
                   args.selection_metric)

    if args.mode in ('final', 'all'):
        run_final(args.method, args.target, seeds, args.data_dir,
                  output_dir, args.max_epochs, args.patience, args.early_stop_metric,
                  args.device, args.verbose, args.dry_run, env_meta,
                  args.selection_metric)

    elapsed = time.time() - t_start
    print(f"\n===== Done. Total wall-time: {elapsed/60:.1f} min =====")


if __name__ == '__main__':
    main()
