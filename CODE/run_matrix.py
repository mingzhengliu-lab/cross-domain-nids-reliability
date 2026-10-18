"""Run the paper's complete coordinate grids, or preview them without training."""
from pathlib import Path
import argparse
import itertools
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('unsw', 'cic', 'bot', 'ton')
METHODS = ('erm', 'dann', 'coral', 'groupdro', 'irm', 'fishr', 'boda')
SEEDS = (2026, 2027, 2028)
SUPPORT_SEEDS = (11, 23, 37, 53, 71)


def commands(kind, reruns, data, device):
    entry = [sys.executable, str(ROOT / 'CODE/run_paper.py')]
    if kind == 'stage3':
        for method, target in itertools.product(METHODS, DOMAINS):
            yield entry + ['stage3', '--method', method, '--target', target, '--mode', 'all',
                           '--data_dir', str(data), '--output_root', str(reruns/'stage3'), '--device', device]
    elif kind == 'anomaly':
        for method, target, seed in itertools.product(('mahalanobis', 'iforest', 'autoencoder', 'deep_svdd'), DOMAINS, SEEDS):
            yield entry + ['anomaly', '--detector', method, '--target', target, '--seed', str(seed),
                           '--data_dir', str(data), '--output_root', str(reruns/'anomaly'), '--device', device]
    elif kind == 'domain-fingerprint':
        for target, rep in itertools.product(DOMAINS, ('raw39', 'relational_v1')):
            for task, model, seed in itertools.product(('all', 'benign', 'attack'), ('logreg', 'mlp'), SEEDS):
                yield entry + ['domain-fingerprint', '--kind', 'domain', '--outer', target,
                               '--representation', rep, '--task', task, '--model', model, '--seed', str(seed),
                               '--data-dir', str(data), '--output-root', str(reruns/'domain_fingerprint'),
                               '--device', 'cpu' if model == 'logreg' else device, '--write-manifest']
            for seed in SEEDS:
                yield entry + ['domain-fingerprint', '--kind', 'model', '--outer', target,
                               '--representation', rep, '--seed', str(seed), '--data-dir', str(data),
                               '--output-root', str(reruns/'domain_fingerprint'), '--device', device, '--write-manifest']
    else:
        passive = kind == 'passive-neural'
        for target, model_seed, support_seed in itertools.product(DOMAINS, SEEDS, SUPPORT_SEEDS):
            yield entry + ['adaptation', '--target', target, '--checkpoint_seed', str(model_seed),
                           '--model_seed', str(model_seed), '--adaptation_seed', str(model_seed),
                           '--support_seed', str(support_seed), '--ns', '100' if passive else '50,100,500,1000',
                           '--modes', 'full_ft,target_only', '--support_policy', 'random_group' if passive else 'balanced',
                           '--checkpoint_template', str(reruns/'stage3/erm_{target}/final/seed{seed}_artifacts/checkpoint.pt'),
                           '--output_root', str(reruns/('passive_neural' if passive else 'adaptation')),
                           '--data_dir', str(data), '--device', device]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('matrix', choices=('stage3', 'anomaly', 'domain-fingerprint', 'adaptation', 'passive-neural'))
    p.add_argument('--reruns-root', type=Path, default=ROOT/'RESULTS/reruns')
    p.add_argument('--data-dir', type=Path, default=ROOT/'DATA/nfv2_processed')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    plan = list(commands(args.matrix, args.reruns_root.resolve(), args.data_dir.resolve(), args.device))
    for index, command in enumerate(plan, 1):
        print(f'[{index}/{len(plan)}] {shlex.join(command)}', flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
