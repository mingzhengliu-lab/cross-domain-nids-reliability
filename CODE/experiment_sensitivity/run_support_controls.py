"""Recompute support-sampling and ToN split sensitivity with fixed source models."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'CODE/common')]
from nidsbed.datasets import LODODataLoader
from nidsbed.data_utils import load_domain_split
from CODE.analysis.output_paths import output_directory
from CODE.experiment_staircase.run_s0_s3_staircase import cache, cdf_fit_transform, METHODS, TARGETS, SUPPORT_SEEDS
from CODE.experiment_sensitivity.common import load_indices, check_groups, group_weights


def frozen_scores(score_root, target):
    columns = []
    for method in METHODS:
        c = cache(score_root, method, target)
        direction = 1 if roc_auc_score(c['source_val_labels'], c['source_val_scores']) >= .5 else -1
        columns.append(direction*cdf_fit_transform(c['source_val_scores'], c['target_scores']))
    return np.column_stack(columns)


def predict_stages(features, scores, labels, support, test):
    """Use support labels only; the caller evaluates test labels afterwards."""
    s0 = scores[test].mean(axis=1)
    if np.unique(labels[support]).size < 2:
        return {'S0': s0, 'S1': s0.copy(), 'S2': s0.copy(), 'S3': np.full(len(test), float(labels[support][0]))}, 0
    signs = np.array([1 if roc_auc_score(labels[support], scores[support, j]) >= .5 else -1 for j in range(scores.shape[1])])
    oriented = scores*signs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        combiner = LogisticRegression(C=.1, solver='lbfgs', max_iter=1000).fit(oriented[support], labels[support])
        probe = LogisticRegression(C=.1, solver='lbfgs', max_iter=1000).fit(features[support], labels[support])
        result = {'S0': s0, 'S1': oriented[test].mean(axis=1),
                  'S2': combiner.predict_proba(oriented[test])[:, 1], 'S3': probe.predict_proba(features[test])[:, 1]}
    return result, sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'DATA/nfv2_processed')
    p.add_argument('--stage3-root', type=Path, default=ROOT/'RESULTS/reruns/stage3')
    p.add_argument('--output-dir', type=Path, default=ROOT/'RESULTS/reruns/support_controls')
    p.add_argument('--targets', nargs='+', choices=TARGETS, default=list(TARGETS))
    p.add_argument('--support-seeds', nargs='+', type=int, choices=SUPPORT_SEEDS, default=list(SUPPORT_SEEDS))
    p.add_argument('--budgets', nargs='+', type=int, choices=(20, 50, 100), default=[20, 50, 100])
    p.add_argument('--sampling', nargs='+', choices=('balanced', 'uniform_group', 'row_arrival'), default=['balanced', 'uniform_group', 'row_arrival'])
    p.add_argument('--skip-alternative-splits', action='store_true')
    p.add_argument('--save-scores', action='store_true')
    args = p.parse_args()
    out = output_directory(args.output_dir)
    indices = load_indices(args.data_dir)
    split_rows = []
    for target in args.targets:
        loader = LODODataLoader(data_dir=str(args.data_dir), target_domain=target, target_few_shot_n=0)
        loader.prepare()
        features, labels = np.asarray(loader._target_X), np.asarray(loader._target_y)
        split = load_domain_split(args.data_dir, target)
        gids = split['row_group_id']
        scores = frozen_scores(args.stage3_root, target)
        if len(scores) != len(features):
            raise ValueError('Stage3 target scores do not align with the fixed data rows')
        specs = [(policy, None, n, seed) for policy in args.sampling for n in args.budgets for seed in args.support_seeds]
        if target == 'ton' and not args.skip_alternative_splits:
            specs += [('balanced', split_seed, 100, seed) for split_seed in range(2029, 2034) for seed in args.support_seeds]
        rows, alternative_rows, freeze = [], [], []
        for policy, split_seed, n, seed in specs:
            prefix = target if split_seed is None else f'ton_split{split_seed}'
            stem = f'{prefix}_{policy}_{seed}_{n}'
            support = indices['support__'+stem]
            train, test = split['train_idx'], split['test_idx']
            if split_seed is not None:
                train, test = indices[f'ton_split{split_seed}__train_idx'], indices[f'ton_split{split_seed}__test_idx']
            test = np.sort(test)
            check_groups(support, train, test, gids)
            if len(support) != n:
                raise ValueError('Support count differs from the requested budget')
            predictions, convergence = predict_stages(features, scores, labels, support, test)
            freeze.append({'coordinate': stem, 'support_sha256': hashlib.sha256(support.tobytes()).hexdigest(),
                           'prediction_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in predictions.items()}})
            if args.save_scores:
                np.savez_compressed(out/(stem+'.npz'), support_idx=support, test_idx=test, **predictions)
            y = labels[test]
            weight = group_weights(gids[test])
            for stage, values in predictions.items():
                record = dict(target=target, support_seed=seed, n_labeled=n, stage=stage, n_support=len(support),
                              support_attack=int(labels[support].sum()), support_test_group_overlap=0,
                              one_class_support=np.unique(labels[support]).size < 2, convergence_warnings=convergence,
                              row_auroc=roc_auc_score(y, values), row_auprc=average_precision_score(y, values),
                              group_auroc=roc_auc_score(y, values, sample_weight=weight),
                              group_auprc=average_precision_score(y, values, sample_weight=weight),
                              n_rows=len(test), n_groups=len(np.unique(gids[test])))
                record['sampling' if split_seed is None else 'split_seed'] = policy if split_seed is None else split_seed
                (rows if split_seed is None else alternative_rows).append(record)
        pd.DataFrame(rows).to_csv(out/f'sampling_{target}.csv', index=False)
        if alternative_rows:
            pd.DataFrame(alternative_rows).to_csv(out/'ton_split_sensitivity.csv', index=False)
            for seed in range(2029, 2034):
                for partition in ('train_idx', 'val_idx', 'test_idx'):
                    ix = indices[f'ton_split{seed}__{partition}']
                    split_rows.append(dict(split_seed=seed, partition=partition, n_rows=len(ix), attack_rate=float(labels[ix].mean()), n_groups=len(np.unique(gids[ix]))))
            pd.DataFrame(split_rows).to_csv(out/'ton_alternative_splits.csv', index=False)
        (out/f'prediction_freeze_{target}.json').write_text(json.dumps({'source_models_fixed': True, 'source_fitted_preprocessing': True, 'coordinates': freeze}, indent=2)+'\n', encoding='utf-8')
        print(f'{target}: {len(rows)} sampling rows and {len(alternative_rows)} alternative-split rows', flush=True)


if __name__ == '__main__':
    main()
