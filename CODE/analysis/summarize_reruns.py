"""Aggregate newly generated experiment records without reading RESULTS_PAPER."""
from pathlib import Path
import argparse
import itertools
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'CODE/common')]
from CODE.analysis.output_paths import output_directory
from CODE.analysis.paper_results import DOMAINS, DISPLAY, SUPPORT_SEEDS
from CODE.experiment_sensitivity.common import group_weights
from nidsbed.data_utils import load_domain_split

SEEDS = (2026, 2027, 2028)
MODES = ('full_ft', 'target_only')
BUDGETS = (50, 100, 500, 1000)


def read_result(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'success':
        raise ValueError(f'Incomplete experiment: {path}')
    if result.get('support_audit', {}).get('support_eval_group_overlap', 0) != 0:
        raise ValueError(f'Support/test overlap: {path}')
    return result


def adapted_folder(root, target, n, mode, seed, support):
    return root/f'{target}/n{n}/{mode}/ckpt{seed}_support{support}_adapt{seed}'


def adaptation(args, out):
    from nidsbed.model_selection import evaluate_score_arrays, select_threshold_at_max_fpr, metrics_at_fixed_threshold
    rows = []
    for target, n, mode, ms, ss in itertools.product(DOMAINS, BUDGETS, MODES, SEEDS, SUPPORT_SEEDS):
        path = adapted_folder(args.adaptation_root, target, n, mode, ms, ss)/'result.json'
        d = read_result(path)
        for key, value in [('target', target), ('n_labeled', n), ('mode', mode), ('checkpoint_seed', ms), ('support_seed', ss)]:
            if d[key] != value:
                raise ValueError(f'Coordinate metadata mismatch in {path}')
        m = d['target_fixed_test']
        source = d['deployment']['source_pooled']['fpr_0.01']['target_fixed_test']
        worst = d['deployment']['source_worst_domain']['fpr_0.01']['target_fixed_test']
        row = dict(result_type='adapted', path=str(path), protocol_version=d['protocol_version'], target=target,
                   n_labeled=n, mode=mode, checkpoint_seed=ms, support_seed=ss, adaptation_seed=d['adaptation_seed'],
                   model_seed=d['model_seed'], auroc=m['auroc'], auprc=m['auprc'], unique_auroc=m['unique_auroc'],
                   unique_auprc=m['unique_auprc'], target_oracle_tpr_at_fpr_1pct=m['tpr_at_fpr_1pct'])
        for prefix, block in [('source_threshold', source), ('source_worst_threshold', worst)]:
            for name in ('fpr', 'tpr', 'precision'):
                row[f'{prefix}_target_{name}_1pct'] = block[name]
            row[f'{prefix}_alerts_per_million_1pct'] = block['alerts_per_million_flows']
        rows.append(row)
    # The zero-shot model is evaluated on the SAME fixed target test population.
    for target, seed in itertools.product(DOMAINS, SEEDS):
        folder = args.stage3_root/f'erm_{target}/final/seed{seed}_artifacts'
        split = load_domain_split(args.data_dir, target)
        test = np.sort(split['test_idx'])
        source_y = np.load(folder/'source_val_labels.npy')
        source_scores = np.load(folder/'source_val_scores.npy')
        threshold = select_threshold_at_max_fpr(source_y, source_scores, .01)['threshold']
        all_scores = np.load(folder/'target_scores.npy')
        all_y = np.load(folder/'target_labels.npy')
        if not np.array_equal(all_y, np.load(args.data_dir/f'{target}_y.npy')):
            raise ValueError('Zero-shot cache rows differ from the fixed target data')
        y, scores = all_y[test], all_scores[test]
        m = evaluate_score_arrays(y, scores, unique_weights=group_weights(split['row_group_id'][test]))
        operating = metrics_at_fixed_threshold(y, scores, threshold)
        rows.append(dict(result_type='zero_shot', path=str(folder), protocol_version='group_fixed_test_adaptation_v5',
                         target=target, n_labeled=0, mode='zero_shot', checkpoint_seed=seed,
                         support_seed=np.nan, adaptation_seed=np.nan, model_seed=seed,
                         auroc=m['auroc'], auprc=m['auprc'], unique_auroc=m['unique_auroc'], unique_auprc=m['unique_auprc'],
                         target_oracle_tpr_at_fpr_1pct=m['tpr_at_fpr_1pct'],
                         source_threshold_target_fpr_1pct=operating['fpr'], source_threshold_target_tpr_1pct=operating['tpr'],
                         source_threshold_target_precision_1pct=operating['precision'],
                         source_threshold_alerts_per_million_1pct=operating['alerts_per_million_flows']))
    pd.DataFrame(rows).to_csv(out/'runs.csv', index=False, float_format='%.10g')
    print(f'{len(rows)} adaptation/zero-shot coordinates')


def passive(args, out):
    rows = []
    balanced = pd.read_csv(args.reruns_root/'adaptation_summary/runs.csv')
    balanced = balanced[balanced.n_labeled == 100].set_index(['target', 'mode', 'checkpoint_seed', 'support_seed'])
    for target, mode, ms, ss in itertools.product(DOMAINS, MODES, SEEDS, SUPPORT_SEEDS):
        path = adapted_folder(args.passive_root/'support_policy=random_group', target, 100, mode, ms, ss)/'result.json'
        d = read_result(path)
        m = d['target_fixed_test']
        source = d['deployment']['source_pooled']['fpr_0.01']['target_fixed_test']
        b = balanced.loc[(target, mode, ms, ss)]
        rows.append(dict(path=str(path), target=target, mode=mode, checkpoint_seed=ms, support_seed=ss,
                         auroc_row=m['auroc'], auroc_group=m['unique_auroc'], auprc_row=m['auprc'],
                         source_tpr_row=source['tpr'], source_fpr_row=source['fpr'], support_group_overlap=0,
                         support_labels_used_for_selection=d['support_audit']['support_selection_uses_labels'],
                         device=d['device'], gpu_name=d.get('gpu_name'), elapsed_sec=d['elapsed_sec'],
                         balanced_auroc_row=b.auroc, balanced_auroc_group=b.unique_auroc,
                         delta_auroc_row=m['auroc']-b.auroc, delta_auroc_group=m['unique_auroc']-b.unique_auroc))
    pd.DataFrame(rows).to_csv(out/'passive_neural_coordinates.csv', index=False)
    print(f'{len(rows)} paired passive-neural coordinates')


def families(args, out):
    rows, seeds = [], []
    methods = {'traditional': ('logreg', 'histgb', 'extratrees', 'lightgbm'),
               'anomaly': ('mahalanobis', 'iforest', 'autoencoder', 'deep_svdd')}
    for category, names in methods.items():
        for method, target in itertools.product(names, DOMAINS):
            weights = group_weights(np.load(args.data_dir/f'{target}_duplicate_group_id.npy'))
            for seed in SEEDS:
                if category == 'traditional':
                    folder = args.tabular_root/f'{method}_{target}/final/seed{seed}_artifacts'
                    path = folder.parent/f'seed{seed}.json'
                    d = read_result(path)
                    record = dict(method=method, target=target, seed=seed, hp_id=d['hp_id'], degradation=d['degradation'])
                    for prefix, block in [('in', d['in_domain']), ('cross', d['cross_domain'])]:
                        for metric in ('attack_f1', 'auroc', 'auprc'):
                            record[prefix+'_'+metric] = block[metric]
                    seeds.append(record)
                else:
                    folder = args.anomaly_root/f'{method}_{target}/seed{seed}_artifacts'
                scores, labels = np.load(folder/'target_scores.npy'), np.load(folder/'target_labels.npy')
                if not np.array_equal(labels, np.load(args.data_dir/f'{target}_y.npy')):
                    raise ValueError(f'Model-family cache row order differs: {folder}')
                rows.append(dict(category=category, method=method, target=target, seed=seed, n_rows=len(labels),
                                 n_groups=int(round(weights.sum())), row_auroc=roc_auc_score(labels, scores),
                                 row_auprc=average_precision_score(labels, scores),
                                 group_auroc=roc_auc_score(labels, scores, sample_weight=weights),
                                 group_auprc=average_precision_score(labels, scores, sample_weight=weights)))
    frame = pd.DataFrame(rows)
    summary = []
    for (category, method, target), block in frame.groupby(['category', 'method', 'target'], sort=False):
        row = dict(category=category, method=method, target=target, target_display=DISPLAY[target], n_seeds=len(block),
                   n_rows=int(block.n_rows.iloc[0]), n_groups=int(block.n_groups.iloc[0]))
        for col in ('row_auroc', 'group_auroc', 'row_auprc', 'group_auprc'):
            row[col+'_mean'], row[col+'_std'] = float(block[col].mean()), float(block[col].std(ddof=1))
        summary.append(row)
    pd.DataFrame(seeds).to_csv(out/'tabular_seed_metrics.csv', index=False)
    pd.DataFrame(summary).to_csv(out/'baseline_results.csv', index=False)
    frame.to_csv(out/'family_seed_metrics.csv', index=False)
    print(f'{len(summary)} model-family summaries')


def staircase(args, out):
    frame = pd.read_csv(args.reruns_root/'staircase/staircase_metrics.csv')
    expected = set(itertools.product(DOMAINS, (20, 50, 100), SUPPORT_SEEDS, ('S0', 'S1', 'S2', 'S3')))
    keys = list(frame[['target', 'n_labeled', 'support_seed', 'stage']].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete staircase coordinate grid')
    rows = []
    for stage, block in frame.groupby('stage', sort=True):
        if stage == 'S0' and not block.groupby('target').auroc_eval_group.nunique().eq(1).all():
            raise ValueError('S0 baselines differ across support coordinates')
        units = block.drop_duplicates('target') if stage == 'S0' else block
        rows.append(dict(stage=stage, row_mean=units.auroc_eval.mean(), row_sd=units.auroc_eval.std(ddof=1),
                         group_mean=units.auroc_eval_group.mean(), group_sd=units.auroc_eval_group.std(ddof=1),
                         n_summary_units=len(units), n_cells=len(block), support_mean=block.support_auroc.mean(),
                         flip_mean=block.n_score_flips.mean()))
    pd.DataFrame(rows).to_csv(out/'staircase_macro.csv', index=False)
    frame.groupby(['target', 'n_labeled', 'stage']).auroc_eval_group.agg(['mean', 'std']).reset_index().to_csv(out/'staircase_by_target_budget.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('adaptation', 'passive', 'families', 'staircase'))
    p.add_argument('--reruns-root', type=Path, default=ROOT/'RESULTS/reruns')
    p.add_argument('--data-dir', type=Path, default=ROOT/'DATA/nfv2_processed')
    for name in ('adaptation', 'stage3', 'passive', 'tabular', 'anomaly'):
        p.add_argument('--'+name+'-root', type=Path)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    for name, directory in [('adaptation', 'adaptation'), ('stage3', 'stage3'), ('passive', 'passive_neural'), ('tabular', 'tabular_baselines'), ('anomaly', 'anomaly')]:
        if getattr(args, name+'_root') is None:
            setattr(args, name+'_root', args.reruns_root/directory)
    out = output_directory(args.output_dir or args.reruns_root/(args.kind+'_summary'))
    globals()[args.kind](args, out)


if __name__ == '__main__':
    main()
