"""Verify the compact evidence chain without raw data, GPUs or prediction caches.

Checks coordinates, paired controls, summary arithmetic, split audits and
archived artifact hashes. Add --with-manuscript to also check a local paper/
directory; the manuscript is not distributed with the code release.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from CODE.analysis.paper_results import (
    ROOT, PAPER, FROZEN, RQ1, RQ2, Q, CONTROL, STAIR, DEPLOY, SENS,
    DOMAINS, METHODS, SEEDS, SUPPORT_SEEDS, read_json, rq1,
    q_coordinates, q_pairs, staircase, deployment, gates,
)

CHECKS = []

def check(name, condition, details=None):
    CHECKS.append(dict(name=name, passed=bool(condition), details=details))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def grid(frame, columns, values):
    actual = list(frame[columns].itertuples(index=False, name=None))
    return len(actual) == len(set(actual)) and set(actual) == set(itertools.product(*values))

def close(left, right, atol=1e-8):
    return np.allclose(left, right, atol=atol, rtol=1e-9, equal_nan=True)

def audit_rq1():
    d = rq1()
    check('RQ1: 28 unique method/target conditions', grid(d, ['method','target'], [METHODS, DOMAINS]))
    errors = []
    pairs = {'in_auroc':'in_auroc', 'cross_auroc':'cross_auroc',
             'cross_unique_auroc':'cross_unique_auroc', 'cross_auprc':'cross_auprc'}
    for row in d.itertuples():
        records = [read_json(RQ1/f'per_cell/{row.method}_{row.target}/seed{s}.json') for s in (2026,2027,2028)]
        for s, record in zip((2026,2027,2028), records):
            if not (record['status']=='success' and record['seed']==s and record['target']==row.target and record['method']==row.method):
                errors.append(f'{row.method}/{row.target}/{s}: invalid coordinate')
        for col, key in pairs.items():
            a = np.array([r[key] for r in records])
            if not close([a.mean(), a.std(ddof=1)], [getattr(row,col+'_mean'),getattr(row,col+'_std')]):
                errors.append(f'{row.method}/{row.target}/{col}: aggregation differs')
    check('RQ1: means and sample SD match 84 seed records', not errors, errors)
    check('RQ1: headline values and below-chance counts',
          close([d.in_auroc_mean.mean(),d.cross_auroc_mean.mean(),d.cross_unique_auroc_mean.mean()],
                [.9464881817,.3348898170,.3846439895],atol=1e-6)
          and (d.cross_auroc_mean<.5).sum()==22 and (d.cross_unique_auroc_mean<.5).sum()==20)
    cache = pd.read_csv(SENS/'rq1_cache_metrics.csv')
    check('RQ1: 84 cached target score coordinates', grid(cache,['method','target','seed'],[METHODS,DOMAINS,(2026,2027,2028)]))
    # The optional verify-data command recomputes these metrics from score arrays.
    means = cache.groupby(['method','target']).target_group_auroc.mean()
    expected = d.set_index(['method','target']).cross_unique_auroc_mean
    check('RQ1: independent cache audit matches target group AUROC', close(means.reindex(expected.index),expected))
    source_overlap = pd.read_csv(SENS/'source_internal_overlap.csv')
    source_sensitivity = pd.read_csv(SENS/'source_internal_sensitivity.csv')
    check('RQ1: pooled-source overlap and 84 exclusion coordinates are complete',
          grid(source_overlap,['target'],[DOMAINS])
          and grid(source_sensitivity,['method','target','seed'],[METHODS,DOMAINS,(2026,2027,2028)])
          and (source_overlap.overlapping_rows < source_overlap.source_test_rows).all())
    idx = ['method','target','seed']
    source_metrics = ['source_row_auroc','source_group_auroc']
    recorded = cache.set_index(idx)[source_metrics]
    recomputed = source_sensitivity.set_index(idx)[source_metrics].reindex(recorded.index)
    check('RQ1: source exclusion preserves the high-source contrast',
          close(recomputed,recorded)
          and close(source_sensitivity[['excluded_source_row_auroc',
                                        'excluded_source_group_auroc']].mean(),
                    [.9698623543,.9704054667],atol=1e-8))

def audit_rq2(with_manuscript=False):
    family_root = RQ2/'EXP03_model_family_controls'
    d = pd.read_csv(family_root/'comparison/baseline_results.csv')
    methods = ('logreg','histgb','extratrees','lightgbm',
               'mahalanobis','iforest','autoencoder','deep_svdd')
    check('RQ2: 32 model-family conditions with three seeds each',
          grid(d,['method','target'],[methods,DOMAINS]) and d.n_seeds.eq(3).all())
    summary = pd.read_csv(family_root/'model_family_summary.csv').set_index('method')
    names = {'HistGB':'HistGradientBoosting','IForest':'IsolationForest','AutoEncoder':'Autoencoder'}
    means = d.groupby('method_display')[['row_auroc_mean','group_auroc_mean']].mean().rename(index=names)
    counts = d.assign(row_below=d.row_auroc_mean<.5,group_below=d.group_auroc_mean<.5).groupby('method_display')[['row_below','group_below']].sum().rename(index=names)
    check('RQ2: family macro means and below-chance counts match archived summary',
          close(means.reindex(summary.index),summary[['row_mean_auroc','group_mean_auroc']],atol=5e-5)
          and close(counts.reindex(summary.index),summary[['row_below_0p5','group_below_0p5']]))
    traditional = d[d.category=='traditional']
    seed_metrics = pd.read_csv(family_root/'traditional_supervised/tabular_seed_metrics.csv')
    raw = pd.read_csv(family_root/'traditional_supervised/tabular_raw.csv')
    check('RQ2: traditional summaries match 48 final seed records and the family comparison',
          grid(seed_metrics,['method','target','seed'],[methods[:4],DOMAINS,(2026,2027,2028)])
          and grid(raw,['method','target'],[methods[:4],DOMAINS])
          and all(close(seed_metrics.groupby(['method','target'])[metric].mean().reindex(
                            raw.set_index(['method','target']).index),raw[metric+'_mean'])
                  for metric in ('cross_attack_f1','cross_auroc','cross_auprc',
                                 'in_attack_f1','in_auroc','in_auprc','degradation'))
          and all(close(seed_metrics.groupby(['method','target'])[metric].std().reindex(
                            raw.set_index(['method','target']).index),raw[metric+'_std'])
                  for metric in ('cross_attack_f1','cross_auroc','cross_auprc'))
          and close(raw.set_index(['method','target']).cross_auroc_mean.reindex(
                        traditional.set_index(['method','target']).index),traditional.row_auroc_mean))
    anomaly = d[d.category=='anomaly']
    operational = pd.read_csv(family_root/'traditional_supervised/tabular_operational.csv')
    prior_resample = pd.read_csv(family_root/'traditional_supervised/tabular_prior_resample.csv')
    check('RQ2: traditional posthoc diagnostics use the same final predictions',
          grid(operational,['method','target','seed'],[methods[:4],DOMAINS,(2026,2027,2028)])
          and grid(prior_resample,['method','target','seed','target_prior'],
                   [methods[:4],DOMAINS,(2026,2027,2028),(.05,.20,.50,.80)])
          and close(operational.set_index(['method','target','seed'])[['f1_at_0p5','auroc','auprc']].reindex(
                        seed_metrics.set_index(['method','target','seed']).index),
                    seed_metrics[['cross_attack_f1','cross_auroc','cross_auprc']]))
    check('RQ2: both model-family blocks retain eight below-chance group conditions',
          len(traditional)==16 and len(anomaly)==16
          and (traditional.group_auroc_mean<.5).sum()==8
          and (anomaly.group_auroc_mean<.5).sum()==8)
    root = RQ2/'EXP05_domain_fingerprint_priorshift'
    domain = pd.read_csv(root/'domain_rows.csv').query("representation=='raw39' and task=='all' and model=='logreg'")
    model = pd.read_csv(root/'model_rows.csv').query("representation=='raw39'")
    check('RQ2: domain fingerprint and independent ERM use all twelve declared coordinates',
          grid(domain,['outer_target','seed'],[DOMAINS,(2026,2027,2028)])
          and grid(model,['outer_target','seed'],[DOMAINS,(2026,2027,2028)]))
    tex = (PAPER/'main.tex').read_text(encoding='utf-8') if with_manuscript else None
    if with_manuscript:
        source = domain.groupby('outer_target').macro_ovr_auroc.mean()
        target = model.groupby('outer_target')[['row_auroc','group_auroc']].mean()
        block = tex.split(r'\label{tab:domain_fingerprint}',1)[1].split(r'\end{table}',1)[0]
        errors = []
        for t in DOMAINS:
            match = re.search(rf'(?m)^{t}\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*(\d+)/3',block,re.I)
            expected = [source.loc[t],target.loc[t,'row_auroc'],target.loc[t,'group_auroc']]
            inverted = int((model.loc[model.outer_target==t,'group_auroc']<.5).sum())
            if not match or not close([float(x) for x in match.groups()[:3]],expected,atol=5e-4) or int(match[4])!=inverted:
                errors.append(t)
        check('RQ2: manuscript fingerprint table matches independent seed records',not errors,errors)
    structure = pd.read_csv(SENS/'data_structure.csv').set_index('target')
    prior = read_json(root/'prior_shift_correlation.json')
    results = rq1()
    errors = []
    block = tex.split(r'\label{tab:prior_correlation}',1)[1].split(r'\end{table}',1)[0] if with_manuscript else None
    for coordinate, numerator, denominator, label in (
            ('row','n_attack_rows','n_rows','Row prior / row AUROC'),
            ('group','n_attack_groups','n_groups','Group prior / group AUROC')):
        rate = structure[numerator] / structure[denominator]
        recorded = prior[f'domain_attack_rates_{coordinate}_level_full_data']
        if not close(rate,[recorded[t] for t in rate.index]):
            errors.append(f'{coordinate}: prior population mismatch')
        x = results.target.map({t:rate[t]-rate.drop(t).mean() for t in DOMAINS})
        for ycoordinate, column in [('row','cross_auroc_mean'),('group','cross_unique_auroc_mean')]:
            r = np.corrcoef(x,results[column])[0,1]
            saved = prior['coordinate_sensitivity'][f'x_{coordinate}__y_{ycoordinate}']
            if not close([r,r*r],[saved['cell_r'],saved['cell_r_squared']]):
                errors.append(f'{coordinate}/{ycoordinate}: correlation mismatch')
            if with_manuscript and coordinate==ycoordinate:
                expected = f'{label} & {r:.3f} & {r*r:.3f} & 28 & 4'
                if expected not in block:
                    errors.append(f'{coordinate}: manuscript rounded values mismatch')
    if any(not (ROOT/p['path']).is_file() or sha(ROOT/p['path'])!=p['sha256'] for p in prior['inputs']):
        errors.append('prior input hash mismatch')
    name = 'RQ2: priors and AUROCs share the retained population; archived correlations agree'
    if with_manuscript:
        name += '; manuscript correlations agree'
    check(name,not errors,errors)


def audit_q(with_manuscript=False):
    d = q_coordinates()
    check('Q: 108 main-arm coordinates on the declared nine seeds',
          grid(d,['target','seed','method'],[DOMAINS,SEEDS,('q_expert_median_inductive','source_erm','native78')]))
    q = pd.DataFrame(read_json(Q/'metrics_long.json'))
    c = pd.DataFrame(read_json(CONTROL/'metrics.json')).rename(columns={'source_seed':'seed'})
    a = q[q.method=='source_erm'].set_index(['target','seed'])
    b = c[c.method=='native78_pooled'].set_index(['target','seed']).reindex(a.index)
    check('Q: source-LR equals the matched pooled native78 control',
          close(a[['row_auroc','group_auroc','row_auprc','group_auprc']],b[['row_auroc','group_auroc','row_auprc','group_auprc']],atol=1e-12))
    bad = []
    for target, seed in itertools.product(DOMAINS,SEEDS):
        a = read_json(Q/f'per_cell/{target}_seed{seed}/prediction_freeze.json')
        b = read_json(CONTROL/f'per_cell/{target}_seed{seed}/prediction_freeze.json')
        if a['target_labels_loaded'] or b['target_labels_loaded'] or a['source_indices_sha256']!=b['source_indices_sha256']:
            bad.append(f'{target}/{seed}')
    check('Q: all 36 pairing and prediction-sealing manifests agree',not bad,bad)
    p = q_pairs()
    if with_manuscript:
        tex = (PAPER/'main.tex').read_text(encoding='utf-8')
        block = tex.split(r'\label{tab:matched_experts}',1)[1].split(r'\end{table}',1)[0]
        errors = []
        for target in DOMAINS:
            row = re.search(rf'(?mi)^{target}\s*&(.+)$',block)
            z = p.loc[target]
            delta = z.q_expert_median_inductive-z.native78
            expected = [z.q_expert_median_inductive.mean(),z.q_expert_median_inductive.std(),
                        z.native78.mean(),z.native78.std(),delta.mean(),delta.std()]
            actual = re.findall(r'[-+]?\d+\.\d+',row[1]) if row else []
            if len(actual)!=6 or [f'{float(v):.3f}' for v in actual]!=[f'{v:.3f}' for v in expected]:
                errors.append(target)
        check('Q: all matched-expert means and paired SDs use one-step three-decimal rounding',not errors,errors)
    delta = (p.q_expert_median_inductive-p.source_erm).groupby('target').mean()
    check('Q: improvement on three domains and negative ToN transfer',
          (delta.loc[['unsw','cic','bot']]>0).all() and delta.loc['ton']<0)


def audit_staircase():
    d = staircase()
    check('S0-S3: complete 240-coordinate grid',grid(d,['target','n_labeled','support_seed','stage'],[DOMAINS,(20,50,100),SUPPORT_SEEDS,('S0','S1','S2','S3')]))
    check('S0-S3: budget counts and fixed test sizes',
          (d.n_support_groups==d.n_labeled).all() and (d.n_support_rows==d.n_labeled).all()
          and d.groupby('target').n_eval.nunique().eq(1).all())
    s0 = d[d.stage=='S0'].groupby('target').auroc_eval_group
    check('S0: four unique baselines, not 60 independent observations',s0.nunique().eq(1).all())
    means = {s:float(d[d.stage==s].auroc_eval_group.mean()) for s in ('S0','S1','S2','S3')}
    check('S0-S3: manuscript stage means',close(list(means.values()),[.368446,.660164,.689228,.905700],atol=2e-6))

def audit_deployment():
    d = deployment()
    raw = pd.read_csv(DEPLOY/'coordinate_rows.csv')
    expected = {(t,n,m) for t in DOMAINS for n in (10,20,50,100,500,1000) for m in ('full_ft','target_only')}
    expected.update((t,0,'zero_shot') for t in DOMAINS)
    check('RQ4: 732 coordinates and 52 unique domain/budget/mode cells',
          len(raw)==732 and len(d)==52 and set(d[['target','n_labeled','mode']].itertuples(index=False,name=None))==expected)
    key = ['target','n_labeled','mode','checkpoint_seed','support_seed']
    check('RQ4: unique coordinates, correct seed counts, no support/test overlap',
          not raw.duplicated(key).any() and raw.support_group_overlap.eq(0).all()
          and (d.n_cells==np.where(d.n_labeled==0,3,15)).all())
    metrics = ['auroc_row','auroc_group','source_threshold_fpr_row','source_threshold_tpr_row','alerts_per_million_row']
    agg = raw.groupby(['target','n_labeled','mode'])[metrics].mean()
    target = d.set_index(['target','n_labeled','mode']).reindex(agg.index)
    # Historical CSVs are rounded to ten significant digits.
    check('RQ4: published means match every seed coordinate',close(agg,target[[m+'_mean' for m in metrics]].to_numpy(),atol=1e-4))
    rates = pd.read_csv(SENS/'data_structure.csv').set_index('target').test_attack_rate
    pi = raw.target.map(rates)
    burden = 1e6*(pi*raw.source_threshold_tpr_row+(1-pi)*raw.source_threshold_fpr_row)
    check('RQ4: alert burden uses all-flow prevalence and row FPR/TPR',close(burden,raw.alerts_per_million_row,atol=.002))
    g = gates(d)
    check('RQ4: 13 distinct configurations, five ranking passes, zero operating passes',
          len(g)==13 and g.rank_pass.sum()==5 and g.operating_pass.sum()==0)
    # Guard the corrected aggregator against silently treating a partial matrix
    # as a four-domain pass or using the worst-source row threshold as a group one.
    from CODE.experiment_low_label.aggregate_nfv2_target_label_budget_confirmation_v1 import _gate_summary
    probe = d[(d.n_labeled==1000)&(d['mode']=='full_ft')].iloc[:1]
    gate = _gate_summary(probe)['budgets']['1000']['full_ft']
    check('Aggregator: one domain cannot satisfy a four-domain ranking gate',not gate['all_targets_row_ge_floor'])

def audit_sensitivity():
    sampling = pd.concat([pd.read_csv(SENS/f'sampling_{t}.csv') for t in DOMAINS],ignore_index=True)
    check('Sampling: 720 coordinates and no support/test overlap',
          grid(sampling,['target','sampling','n_labeled','support_seed','stage'],[DOMAINS,('balanced','uniform_group','row_arrival'),(20,50,100),SUPPORT_SEEDS,('S0','S1','S2','S3')])
          and sampling.support_test_group_overlap.eq(0).all())
    one = sampling[sampling.one_class_support]
    groups = one[['target','sampling','n_labeled','support_seed']].drop_duplicates()
    values = one.pivot(index=['target','sampling','n_labeled','support_seed'],columns='stage',values='group_auroc')
    check('Sampling: five one-class supports produce 20 stage records',len(groups)==5 and len(one)==20
          and close(values.S0,values.S1) and close(values.S0,values.S2) and close(values.S3,.5))
    cal = pd.concat([pd.read_csv(SENS/f'calibration_{t}.csv') for t in DOMAINS],ignore_index=True)
    check('Calibration: 960 complete disjoint coordinates',
          grid(cal,['target','n_labeled','mode','model_seed','support_seed','n_calibration'],[DOMAINS,(50,100,500,1000),('full_ft','target_only'),(2026,2027,2028),SUPPORT_SEEDS,(100,500)])
          and cal.calibration_support_group_overlap.eq(0).all() and cal.calibration_test_group_overlap.eq(0).all())
    check('Calibration: 55% lack resolution for empirical 1% normal FPR',
          cal.calibration_underresolved.eq(cal.calibration_normal<100).all() and close(cal.calibration_underresolved.mean(),.55))
    c = cal[(cal.n_labeled==1000)&(cal['mode']=='full_ft')]
    counts = []
    for n in (100,500):
        z = c[c.n_calibration==n].groupby('target')[['fpr_row','tpr_row']].mean()
        counts.append(int(((z.fpr_row<=.05)&(z.tpr_row>=.2)).sum()))
    check('Calibration: 100/500 extra labels yield four/three passing domains',counts==[4,3])
    ton100 = c[(c.target=='ton')&(c.n_calibration==100)]
    check('Calibration: ToN mean success does not imply every run passes',
          len(ton100)==15
          and int(((ton100.fpr_row<=.05)&(ton100.tpr_row>=.2)).sum())==7
          and ton100.calibration_normal.min()==19
          and ton100.calibration_normal.max()==21
          and (ton100.calibration_fpr==0).all())
    neural = pd.read_csv(SENS/'passive_neural_coordinates.csv')
    check('Passive neural: 120 complete disjoint coordinates',
          grid(neural,['target','mode','checkpoint_seed','support_seed'],[DOMAINS,('full_ft','target_only'),(2026,2027,2028),SUPPORT_SEEDS])
          and neural.support_group_overlap.eq(0).all())
    check('Passive neural: paired differences match recorded baselines',
          close(neural.auroc_row-neural.balanced_auroc_row,neural.delta_auroc_row))

def audit_files():
    provenance = read_json(SENS/'provenance.json')
    bad = [r['distributed'] for r in provenance['artifacts'] if sha(ROOT/r['distributed'])!=r['sha256']]
    check('Sensitivity: compact evidence matches collected artifact hashes',not bad,bad)
    freeze = read_json(DEPLOY/'freeze_manifest.json')
    essential = [r for r in freeze['copied_files'] if not r['path'].endswith('.md')]
    check('Deployment: original frozen evidence hashes remain intact',
          all((DEPLOY/r['path']).is_file() and sha(DEPLOY/r['path'])==r['sha256'] for r in essential))
    failures = []
    for p in (ROOT/'CODE').rglob('*.py'):
        try: ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        except SyntaxError as e: failures.append(f'{p.relative_to(ROOT)}:{e.lineno}: {e.msg}')
    check('Code: every Python source parses',not failures,failures)
    from CODE.run_paper import COMMANDS
    check('CLI: every advertised entry point exists',all((ROOT/p).is_file() for p,_ in COMMANDS.values()))


def audit_manuscript_files():
    tex = (PAPER/'main.tex').read_text(encoding='utf-8')
    tex = re.sub(r'(?<!\\)%[^\n]*', '', tex)
    paths = re.findall(r'\\input\{([^}]+)\}',tex)
    check('Manuscript: every included table exists',all((PAPER/p).is_file() for p in paths))
    texts = tex+'\n'+'\n'.join((PAPER/p).read_text(encoding='utf-8') for p in paths if (PAPER/p).is_file())
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',texts)
    check('Manuscript: nine figures including three editable concepts',len(figures)==9 and all((PAPER/p).is_file() for p in figures))
    figure_manifest = read_json(PAPER/'generated/figure_provenance.json')
    bad = [r['path'] for r in figure_manifest['files'] if not (ROOT/r['path']).is_file() or sha(ROOT/r['path'])!=r['sha256']]
    check('Figures: empirical assets and editable concept exports match recorded hashes',not bad,bad)
    regenerated = [r for r in figure_manifest['files'] if r['role']=='regenerated empirical figure']
    check('Figures: six generated empirical PDFs have matching editable SVG sources',
          len(regenerated)==6 and {r['figure_number'] for r in regenerated}==set(range(4,10))
          and all((ROOT/r['source']).is_file() and sha(ROOT/r['source'])==r['source_sha256']
                  for r in regenerated))
    generator = ROOT/figure_manifest['empirical_generator']
    check('Figures: distributed plotting script and all archived inputs match recorded hashes',
          generator.is_file() and sha(generator)==figure_manifest.get('empirical_generator_current_sha256', figure_manifest['empirical_generator_sha256'])
          and all(r['inputs'] and all((ROOT/v['path']).is_file() and sha(ROOT/v['path'])==v['sha256']
                                     for v in r['inputs']) for r in regenerated))
    comparison = next(r for r in regenerated if r['figure_number']==5)
    check('Figures: source-target comparison uses row-level AUROC at both endpoints',
          comparison['value_columns']==['in_auroc_mean','cross_auroc_mean']
          and (ROOT/comparison['data_source']).is_file()
          and sha(ROOT/comparison['data_source'])==comparison['data_sha256'])
    labels = re.findall(r'\\label\{([^}]+)\}',texts)
    references = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',texts)
    check('Manuscript: labels unique and all cross references resolve',len(labels)==len(set(labels)) and set(references)<=set(labels))
    keys = {k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',texts) for k in block.split(',')}
    bib = (PAPER/'cas-refs.bib').read_text(encoding='utf-8')
    entries = re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib)
    check('Manuscript: bibliography keys unique and all citations resolve',len(entries)==len(set(entries)) and keys<=set(entries),{'citations':len(keys),'missing':sorted(keys-set(entries))})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'RESULTS/reruns/submission_checks/audit.json')
    p.add_argument('--with-manuscript',action='store_true',
                   help='Also check local paper/main.tex, bibliography and figure assets (not distributed).')
    args = p.parse_args()
    if args.with_manuscript and not (PAPER/'main.tex').is_file():
        p.error('--with-manuscript requires a local paper/main.tex; the manuscript is not distributed.')
    functions = [audit_rq1,audit_rq2,audit_q,audit_staircase,audit_deployment,audit_sensitivity,audit_files]
    if args.with_manuscript:
        functions.append(audit_manuscript_files)
    for fn in functions:
        try:
            if fn in (audit_rq2,audit_q):
                fn(with_manuscript=args.with_manuscript)
            else:
                fn()
        except Exception as e: check(fn.__name__,False,f'{type(e).__name__}: {e}')
    passed = all(c['passed'] for c in CHECKS)
    result = dict(passed=passed,checks=CHECKS,
        scope='Compact archived evidence' + (' and local manuscript consistency' if args.with_manuscript else ''),
        known_limits=['Legacy deployment false_alerts_per_million_* fields mean per million benign; legacy v5 group operating fields contain worst-source row thresholds. The paper uses verified row FPR/TPR and all-flow alert burden only.',
                     'Sampling/calibration sensitivity materials comprise archived coordinates, integrity hashes and table derivation.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'{sum(c["passed"] for c in CHECKS)}/{len(CHECKS)} compact evidence checks passed. Report: {args.output}')
    for c in CHECKS:
        if not c['passed']: print('FAIL:',c['name'],c['details'])
    return 0 if passed else 1

if __name__=='__main__': raise SystemExit(main())
