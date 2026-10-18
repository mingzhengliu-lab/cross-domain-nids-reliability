"""Check frozen splits and recompute exact overlap directly from processed arrays.

No split is generated or rewritten. This script is optional for readers who
only have compact frozen results; it requires DATA/nfv2_processed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from CODE.analysis.paper_results import DOMAINS, METHODS
from CODE.common.nidsbed.data_utils import exact_row_group_ids, load_domain_split

def row_keys(x):
    x = np.ascontiguousarray(x)
    return x.view(np.dtype((np.void, x.dtype.itemsize*x.shape[1]))).reshape(-1)

def member(values, sorted_reference):
    i = np.searchsorted(sorted_reference, values)
    valid = i < len(sorted_reference)
    result = np.zeros(len(values), dtype=bool)
    result[valid] = sorted_reference[i[valid]] == values[valid]
    return result

def run(out, data=None, stage3_root=None):
    data = Path(data or ROOT / 'DATA/nfv2_processed')
    stage3_root = Path(stage3_root or ROOT / 'RESULTS/reruns/stage3')
    xs, ys, splits, train_keys = {}, {}, {}, {}
    structures = []
    for domain in DOMAINS:
        xs[domain] = np.load(data/f'{domain}_X.npy', mmap_mode='r')
        ys[domain] = np.load(data/f'{domain}_y.npy')
        sp = splits[domain] = load_domain_split(data, domain)
        gid, counts = exact_row_group_ids(xs[domain])
        if not np.array_equal(gid, sp['row_group_id']):
            raise ValueError(f'{domain}: stored group ids do not match exact feature bytes')
        idx = np.concatenate([sp[k] for k in ('train_idx','val_idx','test_idx')])
        if not np.array_equal(np.sort(idx),np.arange(len(gid))):
            raise ValueError(f'{domain}: split is not a disjoint partition')
        parts = [np.unique(gid[sp[k]]) for k in ('train_idx','val_idx','test_idx')]
        if any(np.intersect1d(parts[i],parts[j]).size for i in range(3) for j in range(i)):
            raise ValueError(f'{domain}: a duplicate group crosses split boundaries')
        sums = np.bincount(gid, weights=ys[domain])
        if np.any((sums>0)&(sums<counts)):
            raise ValueError(f'{domain}: mixed binary labels within duplicate groups')
        keys = row_keys(xs[domain])
        tr = sp['train_idx']
        train_keys[domain] = {
            label: np.unique(keys[tr[ys[domain][tr]==label]]) for label in (0,1)
        }
        structures.append(dict(target=domain,n_rows=len(gid),n_groups=len(counts),
            n_attack_rows=int(ys[domain].sum()),n_attack_groups=int((sums>0).sum()),
            test_rows=len(sp['test_idx']),test_groups=len(parts[2]),
            train_attack_rate=float(ys[domain][sp['train_idx']].mean()),
            val_attack_rate=float(ys[domain][sp['val_idx']].mean()),
            test_attack_rate=float(ys[domain][sp['test_idx']].mean()),
            mixed_groups=0,split_group_overlap=0))
        print(f'{domain}: validated rows, groups, labels and three-way split',flush=True)
    overlap, sensitivity, cache_checks = [], [], []
    source_overlap, source_sensitivity = [], []
    for target in DOMAINS:
        sp, y = splits[target], ys[target]
        keys = row_keys(xs[target])
        gid = sp['row_group_id']
        source_keys = {label:np.unique(np.concatenate([train_keys[d][label] for d in DOMAINS if d!=target])) for label in (0,1)}
        match0, match1 = member(keys,source_keys[0]),member(keys,source_keys[1])
        any_match = match0 | match1
        opposite = np.where(y==1,match0,match1)
        te = sp['test_idx']
        first = np.unique(gid[te],return_index=True)[1]
        overlap.append(dict(target=target,test_row_any_match=float(any_match[te].mean()),
            test_group_any_match=float(any_match[te][first].mean()),
            test_row_opposite_label=float(opposite[te].mean()),
            test_group_opposite_label=float(opposite[te][first].mean()),
            full_row_any_match=float(any_match.mean()),full_group_any_match=float(any_match[np.unique(gid,return_index=True)[1]].mean())))
        keep = ~any_match
        counts = np.bincount(gid)
        weights = 1/counts[gid]
        # Domain-local splits do not rule out identical feature patterns in
        # another source domain's training partition. Check the pooled source
        # internal test separately, preserving labels in cross-source conflicts.
        sources = [d for d in DOMAINS if d != target]
        source_test_keys = np.concatenate([
            row_keys(xs[d][splits[d]['test_idx']]) for d in sources])
        source_test_y = np.concatenate([
            ys[d][splits[d]['test_idx']] for d in sources])
        _, source_gid, source_counts = np.unique(
            source_test_keys, return_inverse=True, return_counts=True)
        source_positive = np.bincount(source_gid, weights=source_test_y)
        source_mixed = (source_positive > 0) & (source_positive < source_counts)
        source_seen = (member(source_test_keys, source_keys[0]) |
                       member(source_test_keys, source_keys[1]))
        source_keep = ~source_seen
        source_overlap.append(dict(target=target,
            source_test_rows=len(source_test_y), source_test_groups=len(source_counts),
            mixed_label_groups=int(source_mixed.sum()),
            mixed_label_rows=int(source_counts[source_mixed].sum()),
            overlapping_rows=int(source_seen.sum()),
            overlapping_groups=int(np.unique(source_gid[source_seen]).size)))
        for method in METHODS:
            for seed in (2026,2027,2028):
                base = stage3_root/f'{method}_{target}/final/seed{seed}_artifacts'
                scores = np.load(base/'target_scores.npy')
                cached_y = np.load(base/'target_labels.npy')
                if not np.array_equal(cached_y,y): raise ValueError(f'{base}: cached target order mismatch')
                original_row = roc_auc_score(y,scores)
                original_group = roc_auc_score(y,scores,sample_weight=weights)
                sensitivity.append(dict(target=target,method=method,seed=seed,
                    row_auroc=original_row,group_auroc=original_group,
                    excluded_row_auroc=roc_auc_score(y[keep],scores[keep]),
                    excluded_group_auroc=roc_auc_score(y[keep],scores[keep],sample_weight=weights[keep])))
                sy=np.load(base/'source_intest_labels.npy'); ss=np.load(base/'source_intest_scores.npy')
                sw=np.load(base/'source_intest_unique_weights.npy')
                if not np.array_equal(sy, source_test_y):
                    raise ValueError(f'{base}: cached source order mismatch')
                expected_sw = (1/source_counts[source_gid]).astype(np.float32)
                if not np.array_equal(sw, expected_sw):
                    raise ValueError(f'{base}: cached source group weights mismatch')
                source_sensitivity.append(dict(target=target, method=method, seed=seed,
                    source_row_auroc=roc_auc_score(sy,ss),
                    source_group_auroc=roc_auc_score(sy,ss,sample_weight=sw),
                    excluded_source_row_auroc=roc_auc_score(sy[source_keep],ss[source_keep]),
                    excluded_source_group_auroc=roc_auc_score(
                        sy[source_keep],ss[source_keep],sample_weight=sw[source_keep])))
                cache_checks.append(dict(target=target,method=method,seed=seed,
                    source_row_auroc=roc_auc_score(sy,ss),source_group_auroc=roc_auc_score(sy,ss,sample_weight=sw),
                    source_row_auprc=__import__('sklearn.metrics',fromlist=['average_precision_score']).average_precision_score(sy,ss),
                    target_row_auroc=original_row,target_group_auroc=original_group))
        print(f'{target}: recomputed overlap and 21 source/target prediction coordinates',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    for filename,records in [('data_structure.csv',structures),('exact_overlap.csv',overlap),
                             ('overlap_sensitivity.csv',sensitivity),('rq1_cache_metrics.csv',cache_checks),
                             ('source_internal_overlap.csv',source_overlap),
                             ('source_internal_sensitivity.csv',source_sensitivity)]:
        pd.DataFrame(records).to_csv(out/filename,index=False)
    pd.DataFrame(overlap).to_csv(sys.stdout,index=False)
    (out/'manifest.json').write_text(json.dumps(dict(scope='exact float32 feature bytes; source training partitions only',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_role='processed arrays and frozen prediction caches',data_written=False,
        split_overlap=0,coordinates_recomputed=84,
        source_internal_overlap_checked=True),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'RESULTS/reruns/submission_checks/data')
    p.add_argument('--data-dir',type=Path,default=ROOT/'DATA/nfv2_processed')
    p.add_argument('--stage3-root',type=Path,default=ROOT/'RESULTS/reruns/stage3')
    args=p.parse_args()
    run(args.output, args.data_dir, args.stage3_root)
