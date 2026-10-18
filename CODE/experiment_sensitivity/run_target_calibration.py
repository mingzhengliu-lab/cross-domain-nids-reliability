"""Select thresholds from independent target calibration groups; never train models."""
from pathlib import Path
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'CODE/common')]
from nidsbed.algorithms.erm import ERM
from nidsbed.adaptation import model_state_hashes
from nidsbed.datasets import LODODataLoader
from nidsbed.data_utils import load_domain_split
from nidsbed.calibration import scores_from_logits
from nidsbed.model_selection import predict_logits_batched, select_threshold_at_max_fpr, metrics_at_fixed_threshold
from CODE.analysis.output_paths import output_directory
from CODE.experiment_sensitivity.common import load_indices, check_groups, group_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'DATA/nfv2_processed')
    p.add_argument('--adaptation-root', type=Path, default=ROOT/'RESULTS/reruns/adaptation')
    p.add_argument('--output-dir', type=Path, default=ROOT/'RESULTS/reruns/calibration')
    p.add_argument('--targets', nargs='+', choices=('unsw', 'cic', 'bot', 'ton'), default=['unsw', 'cic', 'bot', 'ton'])
    p.add_argument('--budgets', type=int, nargs='+', choices=(50, 100, 500, 1000), default=[50, 100, 500, 1000])
    p.add_argument('--model-seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    p.add_argument('--support-seeds', type=int, nargs='+', default=[11, 23, 37, 53, 71])
    p.add_argument('--modes', nargs='+', choices=('full_ft', 'target_only'), default=['full_ft', 'target_only'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--verify-test-logits', action='store_true')
    args = p.parse_args()
    out = output_directory(args.output_dir)
    indices = load_indices(args.data_dir)
    for target in args.targets:
        loader = LODODataLoader(data_dir=str(args.data_dir), target_domain=target, target_few_shot_n=0)
        loader.prepare()
        features, labels = np.asarray(loader._target_X), np.asarray(loader._target_y)
        split = load_domain_split(args.data_dir, target)
        test = np.sort(split['test_idx'])
        gids = split['row_group_id']
        weights = group_weights(gids[test])
        rows = []
        for n, mode, ms, ss in itertools.product(args.budgets, args.modes, args.model_seeds, args.support_seeds):
            folder = args.adaptation_root/f'{target}/n{n}/{mode}/ckpt{ms}_support{ss}_adapt{ms}'
            result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if result['status'] != 'success':
                raise ValueError(f'Incomplete adaptation result: {folder}')
            with np.load(folder/'support_split.npz') as z:
                support = z['selected_indices']
            calibration = indices[f'calibration__{target}_{n}_{ss}']
            check_groups(calibration, split['train_idx'], test, gids)
            if np.intersect1d(gids[calibration], gids[support]).size:
                raise ValueError('Calibration overlaps adaptation support groups')
            checkpoint = torch.load(folder/'adapted_checkpoint.pt', map_location=args.device, weights_only=True)
            model = ERM(in_dim=int(checkpoint['in_dim']), n_classes=int(checkpoint['n_classes']),
                        hparams=checkpoint['hparams'], class_weights_per_domain={}, device=args.device)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
            before = model_state_hashes(model)
            if before != result['state_hashes']['final']:
                raise ValueError('Adapted model differs from its result record')
            scores = scores_from_logits(predict_logits_batched(model, features[calibration], device=args.device), temperature=1.)
            with np.load(folder/'predictions.npz') as z:
                target_scores = z['target_eval_scores'].copy()
                if not np.array_equal(z['target_eval_labels'], labels[test]):
                    raise ValueError('Frozen evaluation predictions use a different test partition')
                logit_error = None
                if args.verify_test_logits:
                    logits = predict_logits_batched(model, features[test], device=args.device)
                    logit_error = float(np.max(np.abs(logits-z['target_eval_logits'])))
            if before != model_state_hashes(model):
                raise ValueError('Model state changed during calibration inference')
            source = result['deployment']['source_pooled']['fpr_0.01']['target_fixed_test']
            for nc in (100, 500):
                ix, sc = calibration[:nc], scores[:nc]
                selection = select_threshold_at_max_fpr(labels[ix], sc, .01)
                threshold = selection['threshold']
                row = metrics_at_fixed_threshold(labels[test], target_scores, threshold)
                predicted = np.asarray(target_scores, dtype=np.float64) >= threshold
                neg, pos = labels[test] == 0, labels[test] == 1
                rows.append(dict(target=target, n_labeled=n, mode=mode, model_seed=ms, support_seed=ss,
                                 n_calibration=nc, total_label_budget=n+nc, threshold=threshold,
                                 calibration_normal=int((labels[ix] == 0).sum()), calibration_underresolved=selection['underresolved'],
                                 calibration_fpr=selection['reference_fpr'], fpr_row=row['fpr'], tpr_row=row['tpr'],
                                 fpr_group=float(weights[neg & predicted].sum()/weights[neg].sum()),
                                 tpr_group=float(weights[pos & predicted].sum()/weights[pos].sum()),
                                 source_fpr_row=source['fpr'], source_tpr_row=source['tpr'],
                                 calibration_support_group_overlap=0, calibration_test_group_overlap=0,
                                 max_cache_logit_error=logit_error))
        pd.DataFrame(rows).to_csv(out/f'calibration_{target}.csv', index=False)
        print(f'{target}: {len(rows)} calibration conditions; model weights fixed', flush=True)


if __name__ == '__main__':
    main()
