"""
scripts/run_anomaly_benchmark.py

Train a source-benign-only anomaly detector under strict P0 LODO protocol
and evaluate on (source val, source in-test, target cross-test).

Protocol:
  * Per source domain: stratified 80/10/10 split with random_state=2026.
  * StandardScaler fit on concat(3 source train splits); transform all; clip [-10, 10].
  * Detector trains on source TRAIN BENIGN (y=0) ONLY. No attack labels at fit time.
  * Threshold rules:
      - tau_p95 / tau_p99: q-quantile of source train benign scores (P0-clean).
      - tau_oracle: best F1 on target via grid search. This is an upper bound.
  * Primary metric: AUROC (threshold-free); F1 reported at all 3 thresholds.

Output: scratch/exp03_model_family_controls/anomaly_runs/<detector>_<target>/seed{N}.json

Usage:
    python scripts/run_anomaly_benchmark.py --detector mahalanobis --target unsw --seed 42
    python scripts/run_anomaly_benchmark.py --detector deep_svdd --target ton --seed 123 --device cuda
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = PROJECT_ROOT / 'CODE' / 'common'
for import_root in (COMMON_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve,
)

from nidsbed.anomaly import DETECTOR_REGISTRY
from nidsbed.datasets import LODODataLoader


# Number of (fpr, tpr) points to retain per ROC curve in JSON output.
# 256 points give visually smooth curves while keeping JSON size <10KB/run.
ROC_DOWNSAMPLE_POINTS = 256


# -----------------------------------------------------------------------------
# Constants for the shared LODO protocol
# -----------------------------------------------------------------------------
DOMAINS = ['unsw', 'cic', 'bot', 'ton']
LODO_SPLIT_SEED = 2026
OUTPUT_ROOT = PROJECT_ROOT / 'RESULTS' / 'model_family' / 'anomaly'


# -----------------------------------------------------------------------------
# Runtime metadata
# -----------------------------------------------------------------------------
def env_meta():
    """Collect runtime metadata only. No version-control dependency is used."""
    gpu = None
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return {
        'python_version': platform.python_version(),
        'torch_version':  torch.__version__,
        'cuda_version':   torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_name':       gpu,
        'hostname':       socket.gethostname(),
        'platform':       platform.platform(),
        'timestamp':      datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


# -----------------------------------------------------------------------------
# Data loading for the shared LODO protocol
# -----------------------------------------------------------------------------
def load_lodo_split(target: str, data_dir: str):
    """Load and scale LODO data for given target.

    Returns dict with all splits, scaled and clipped.
    """
    loader = LODODataLoader(data_dir=data_dir, target_domain=target)
    loader.prepare()
    X_train, y_train = loader.get_source_train()
    X_val, y_val = loader.get_source_val()
    X_intst, y_intst = loader.get_in_domain_eval()
    X_tgt, y_tgt = loader.get_cross_domain_eval()

    return dict(
        X_src_train=X_train.astype(np.float32, copy=False), y_src_train=y_train,
        X_src_val=X_val.astype(np.float32, copy=False), y_src_val=y_val,
        X_src_intest=X_intst.astype(np.float32, copy=False), y_src_intest=y_intst,
        X_tgt=X_tgt.astype(np.float32, copy=False), y_tgt=y_tgt,
        source_domains=loader.source_domains,
        target=target,
        split_policy=loader.split_audit.get('policy'),
        feature_preprocessing_policy=loader.preprocessing_audit.get('policy'),
    )


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def metrics_at_threshold(y_true, scores, tau):
    pred = (scores >= tau).astype(np.int64)
    return {
        'threshold':    float(tau),
        'attack_f1':    float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        'precision':    float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        'recall':       float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        'positive_rate': float((pred == 1).mean()),
    }


def auroc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def auprc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def tpr_at_fpr(y_true, scores, fpr_target: float):
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    if fpr_target <= fpr[0]:
        return float(tpr[0])
    if fpr_target >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(fpr_target, fpr, tpr))


def roc_downsampled(y_true, scores, n_points: int = ROC_DOWNSAMPLE_POINTS):
    """Compute ROC curve and downsample to n_points along FPR axis.

    Returns dict with keys fpr, tpr (lists, length ~n_points + 2 endpoints).
    sklearn's roc_curve returns up to N+1 points (one per unique threshold);
    for N=500K target rows that's too large to keep in JSON. We linearly
    interpolate TPR at n_points evenly-spaced FPR values.
    """
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Ensure strictly monotone (roc_curve already returns sorted fpr but
    # may have flat segments); np.interp handles flat segments fine.
    fpr_grid = np.linspace(0.0, 1.0, n_points)
    tpr_grid = np.interp(fpr_grid, fpr, tpr)
    return {
        'fpr': [float(x) for x in fpr_grid],
        'tpr': [float(x) for x in tpr_grid],
    }


def best_f1_threshold(y_true, scores, n_grid: int = 199):
    """Sweep tau over score quantiles to find max F1 (oracle upper bound)."""
    qs = np.linspace(0.005, 0.995, n_grid)
    taus = np.quantile(scores, qs)
    best = {'attack_f1': -1.0}
    for tau in taus:
        m = metrics_at_threshold(y_true, scores, tau)
        if m['attack_f1'] > best['attack_f1']:
            best = m
    return best


# -----------------------------------------------------------------------------
# Single run
# -----------------------------------------------------------------------------
def run_one(detector_name: str, target: str, seed: int,
            data_dir: str, device: str = 'cuda', verbose: bool = True,
            detector_kwargs: dict | None = None,
            artifact_dir: Path | None = None) -> dict:
    t0 = time.time()

    if verbose:
        print(f"[load] LODO split target={target}")
    data = load_lodo_split(target, data_dir=data_dir)

    benign_mask = (data['y_src_train'] == 0)
    X_train_benign = data['X_src_train'][benign_mask]
    if verbose:
        print(f"  src train  : {len(data['X_src_train']):>8} total, "
              f"{benign_mask.sum():>8} benign ({benign_mask.mean()*100:.2f}%)")
        print(f"  src val    : {len(data['X_src_val']):>8}")
        print(f"  src intest : {len(data['X_src_intest']):>8}")
        print(f"  target     : {len(data['X_tgt']):>8}  "
              f"attack rate={data['y_tgt'].mean()*100:.2f}%")

    # Build & fit detector
    cls = DETECTOR_REGISTRY[detector_name]
    detector_kwargs = detector_kwargs or {}
    det = cls(random_state=seed, device=device, **detector_kwargs)
    if verbose:
        print(f"[fit] {detector_name} on {len(X_train_benign)} benign samples")
    t_fit = time.time()
    det.fit(X_train_benign)
    fit_sec = time.time() - t_fit
    if verbose:
        print(f"  fit done in {fit_sec:.1f}s")

    # Score
    if verbose:
        print(f"[score]")
    s_train_benign = det.score(X_train_benign)
    s_val          = det.score(data['X_src_val'])
    s_intest       = det.score(data['X_src_intest'])
    s_target       = det.score(data['X_tgt'])

    # Replace NaN/Inf with median of finite scores
    for nm, s in [('train', s_train_benign), ('val', s_val), ('intest', s_intest), ('target', s_target)]:
        bad = ~np.isfinite(s)
        if bad.any():
            med = float(np.nanmedian(s[~bad])) if (~bad).any() else 0.0
            print(f"  [warn] {nm}: {bad.sum()} non-finite scores -> median={med}")
            s[bad] = med

    # Threshold rules from source benign training scores
    tau_p95 = float(np.quantile(s_train_benign, 0.95))
    tau_p99 = float(np.quantile(s_train_benign, 0.99))

    in_metrics = {
        'auroc':   auroc(data['y_src_intest'], s_intest),
        'auprc':   auprc(data['y_src_intest'], s_intest),
        'tpr_at_fpr_1pct': tpr_at_fpr(data['y_src_intest'], s_intest, 0.01),
        'tau_p95': metrics_at_threshold(data['y_src_intest'], s_intest, tau_p95),
        'tau_p99': metrics_at_threshold(data['y_src_intest'], s_intest, tau_p99),
        'roc':     roc_downsampled(data['y_src_intest'], s_intest),
    }
    val_metrics = {
        'auroc':   auroc(data['y_src_val'], s_val),
        'auprc':   auprc(data['y_src_val'], s_val),
        'tpr_at_fpr_1pct': tpr_at_fpr(data['y_src_val'], s_val, 0.01),
        'roc':     roc_downsampled(data['y_src_val'], s_val),
    }
    cross_metrics = {
        'auroc':      auroc(data['y_tgt'], s_target),
        'auprc':      auprc(data['y_tgt'], s_target),
        'tpr_at_fpr_1pct': tpr_at_fpr(data['y_tgt'], s_target, 0.01),
        'tau_p95':    metrics_at_threshold(data['y_tgt'], s_target, tau_p95),
        'tau_p99':    metrics_at_threshold(data['y_tgt'], s_target, tau_p99),
        'tau_oracle': best_f1_threshold(data['y_tgt'], s_target),
        'roc':        roc_downsampled(data['y_tgt'], s_target),
    }

    elapsed = time.time() - t0
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / 'target_scores.npy', s_target.astype(np.float32))
        np.save(artifact_dir / 'target_labels.npy', data['y_tgt'].astype(np.int32))
        np.save(artifact_dir / 'source_intest_scores.npy', s_intest.astype(np.float32))
        np.save(artifact_dir / 'source_intest_labels.npy', data['y_src_intest'].astype(np.int32))
        np.save(artifact_dir / 'source_val_scores.npy', s_val.astype(np.float32))
        np.save(artifact_dir / 'source_val_labels.npy', data['y_src_val'].astype(np.int32))
        np.save(artifact_dir / 'train_benign_scores.npy', s_train_benign.astype(np.float32))
    if verbose:
        a_in = in_metrics['auroc']; a_cr = cross_metrics['auroc']
        print(f"  in_AUROC={a_in:.4f}  cross_AUROC={a_cr:.4f}")
        print(f"  val_AUROC={val_metrics['auroc']:.4f}  cross_TPR@1%={cross_metrics['tpr_at_fpr_1pct']:.4f}")
        print(f"  cross_F1: p95={cross_metrics['tau_p95']['attack_f1']:.4f}  "
              f"p99={cross_metrics['tau_p99']['attack_f1']:.4f}  "
              f"oracle={cross_metrics['tau_oracle']['attack_f1']:.4f}")

    return {
        'detector':            detector_name,
        'target':              target,
        'seed':                seed,
        'fit_sec':             fit_sec,
        'wall_sec':            elapsed,
        'n_train_benign':      int(benign_mask.sum()),
        'n_target':            int(len(data['y_tgt'])),
        'target_attack_rate':  float(data['y_tgt'].mean()),
        'intest_attack_rate':  float(data['y_src_intest'].mean()),
        'split_policy':        data.get('split_policy'),
        'feature_preprocessing_policy': data.get('feature_preprocessing_policy'),
        'detector_kwargs':     detector_kwargs,
        'val':                 val_metrics,
        'in':                  in_metrics,
        'cross':               cross_metrics,
        'env':                 env_meta(),
    }


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def parse_detector_kwargs(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Backward compatibility for older launchers that passed a Python dict
        # literal like "{'n_jobs': 10}" instead of strict JSON.
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise SystemExit(
                "Invalid --detector_kwargs_json. Expected a JSON object like "
                '\'{"n_jobs": 10}\''
            ) from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--detector_kwargs_json must decode to an object/dict.")
    return parsed


def existing_record_complete(path: Path, detector: str, target: str, seed: int) -> bool:
    """Skip only complete, matching anomaly result JSONs."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if rec.get('detector') != detector or rec.get('target') != target or int(rec.get('seed', -1)) != seed:
        return False
    cross = rec.get('cross') or {}
    in_metrics = rec.get('in') or {}
    artifact_dir = path.parent / f'seed{seed}_artifacts'
    required_artifacts = [
        artifact_dir / 'target_scores.npy',
        artifact_dir / 'target_labels.npy',
        artifact_dir / 'source_intest_scores.npy',
        artifact_dir / 'source_intest_labels.npy',
        artifact_dir / 'train_benign_scores.npy',
    ]
    return (
        cross.get('auroc') is not None
        and cross.get('auprc') is not None
        and cross.get('roc') is not None
        and cross.get('tau_p95') is not None
        and cross.get('tau_p99') is not None
        and cross.get('tau_oracle') is not None
        and in_metrics.get('auroc') is not None
        and rec.get('n_target', 0) > 0
        and all(p.exists() and p.stat().st_size > 0 for p in required_artifacts)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', required=True, choices=list(DETECTOR_REGISTRY))
    parser.add_argument('--target',   required=True, choices=DOMAINS)
    parser.add_argument('--seed',     type=int, required=True)
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', default=str(PROJECT_ROOT / 'DATA' / 'nfv2_processed'))
    parser.add_argument('--output_root', default=str(OUTPUT_ROOT))
    parser.add_argument('--detector_kwargs_json', default=None,
                        help='JSON object of kwargs passed to the detector constructor')
    parser.add_argument('--force',    action='store_true', help='overwrite if seed JSON exists')
    parser.add_argument('--quiet',    action='store_true')
    args = parser.parse_args()

    out_dir  = Path(args.output_root) / f'{args.detector}_{args.target}'
    out_path = out_dir / f'seed{args.seed}.json'

    if out_path.exists() and not args.force:
        if existing_record_complete(out_path, args.detector, args.target, args.seed):
            print(f"[skip] complete result exists: {out_path}")
            return
        print(f"[repair] {out_path} exists but is incomplete or mismatched; recomputing")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print(f"Anomaly run - detector={args.detector}  target={args.target}  seed={args.seed}")
    print("=" * 70)

    detector_kwargs = parse_detector_kwargs(args.detector_kwargs_json)
    artifact_dir = out_dir / f'seed{args.seed}_artifacts'
    rec = run_one(args.detector, args.target, args.seed,
                  data_dir=args.data_dir, device=args.device,
                  verbose=not args.quiet, detector_kwargs=detector_kwargs,
                  artifact_dir=artifact_dir)

    atomic_write_json(out_path, rec)
    print(f"[save] {out_path}")
    print(f"[done] wall={rec['wall_sec']:.1f}s  fit={rec['fit_sec']:.1f}s")


if __name__ == '__main__':
    main()
