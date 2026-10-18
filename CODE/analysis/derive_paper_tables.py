"""Derive eight supplementary table summaries and two statistical summaries.

Reads frozen coordinates; writes only to a separate output directory. The
current manuscript contains its own inline tables, checked by check-manuscript.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from CODE.analysis.paper_results import (
    RQ1, RQ2, SENS, STAIR, DOMAINS, DISPLAY,
    rq1, staircase, gates, bootstrap_mean, prior_correlations, tabular_raw,
)
from CODE.analysis.output_paths import output_directory
from CODE.analysis import paper_results as results

OUTPUT_ROOT = ROOT / "RESULTS/reruns/tables"
INPUTS = set()
OUTPUTS = []

def csv(path):
    INPUTS.add(path)
    return pd.read_csv(path)

def table(name,caption,label,columns,rows,compact=False):
    lines = [r'\begin{table}[htbp]',r'\centering\small']
    if compact: lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines += [r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule']
    lines += [r if isinstance(r,str) else ' & '.join(str(c) for c in r)+r' \\' for r in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    p = OUTPUT_ROOT/name
    p.write_text('% Derived by CODE/analysis/derive_paper_tables.py from archived results.\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    OUTPUTS.append(p)

def main():
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'RESULTS_PAPER')
    args = parser.parse_args()
    results.configure_results(args.results_root)
    globals().update({name: getattr(results, name) for name in ('RQ1', 'RQ2', 'SENS', 'STAIR')})
    try:
        OUTPUT_ROOT = output_directory(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    d=csv(RQ1/'Table_3_raw.csv'); cache=csv(SENS/'rq1_cache_metrics.csv')
    prior = prior_correlations(csv(SENS/'data_structure.csv'), d)
    prior['inputs'] = [dict(path=results.input_path(p),
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in (SENS/'data_structure.csv', RQ1/'Table_3_raw.csv')]
    path = OUTPUT_ROOT/'prior_shift_correlation.json'
    path.write_text(json.dumps(prior,indent=2)+'\n',encoding='utf-8')
    OUTPUTS.append(path)
    traditional = RQ2/'EXP03_model_family_controls/traditional_supervised'
    raw_path = OUTPUT_ROOT/'tabular_raw.csv'
    tabular_raw(csv(traditional/'tabular_seed_metrics.csv')).to_csv(raw_path,index=False)
    OUTPUTS.append(raw_path)
    rows=[['Target','Source row AUROC','Target row AUROC','Source group AUROC','Target group AUROC','Target row AUPRC'],r'\midrule']
    for t in DOMAINS:
        z=d[d.target==t];c=cache[cache.target==t]
        rows.append([DISPLAY[t],f'{z.in_auroc_mean.mean():.4f}',f'{z.cross_auroc_mean.mean():.4f}',
                     f'{c.source_group_auroc.mean():.4f}',f'{z.cross_unique_auroc_mean.mean():.4f}',f'{z.cross_auprc_mean.mean():.4f}'])
    table('rq1_matched.tex','Source-internal and target AUROC under row and duplicate-group weighting. Values are derived from archived RQ1 and source-score records.','tab:rq1_matched','lrrrrr',rows)
    family=csv(RQ2/'EXP03_model_family_controls/comparison/baseline_results.csv')
    rows=[['Method','UNSW group','CIC group','BoT group','ToN group','Row macro','Group macro'],r'\midrule']
    family_names={'logreg':'LogReg','histgb':'HistGB','extratrees':'ExtraTrees','lightgbm':'LightGBM',
                  'mahalanobis':'Mahalanobis','iforest':'IsolationForest','autoencoder':'Autoencoder','deep_svdd':'DeepSVDD'}
    for category,title in [('traditional','Source-supervised classifiers'),('anomaly','Source-benign anomaly detectors')]:
        rows.append(r'\multicolumn{7}{l}{\textit{'+title+r'}} \\')
        for method in family.loc[family.category==category,'method'].drop_duplicates():
            z=family[(family.category==category)&(family.method==method)].set_index('target').loc[list(DOMAINS)]
            rows.append([family_names[method],*[f'{v:.4f}' for v in z.group_auroc_mean],
                         f'{z.row_auroc_mean.mean():.4f}',f'{z.group_auroc_mean.mean():.4f}'])
        if category=='traditional': rows.append(r'\midrule')
    table('cross_family_baselines.tex','Model-family controls on all retained target records. Domain values average three training seeds; macro averages weight the four domains equally.','tab:cross_family_baselines','lrrrrrr',rows,compact=True)
    overlap=csv(SENS/'exact_overlap.csv'); sensitivity=csv(SENS/'overlap_sensitivity.csv')
    rows=[['Target','Test row overlap','Test row label conflict','Test group label conflict','All-row overlap','Filtered row AUROC','Filtered group AUROC'],r'\midrule']
    for t in DOMAINS:
        z=overlap[overlap.target==t].iloc[0];s=sensitivity[sensitivity.target==t]
        rows.append([DISPLAY[t],f'{100*z.test_row_any_match:.3f}',f'{100*z.test_row_opposite_label:.3f}',
                     f'{100*z.test_group_opposite_label:.3f}',f'{100*z.full_row_any_match:.3f}',
                     f'{s.excluded_row_auroc.mean():.4f}',f'{s.excluded_group_auroc.mean():.4f}'])
    table('overlap_sensitivity.tex','Exact feature overlap with source training data and removal sensitivity. Overlap fractions are percentages; the last two columns exclude overlapping groups from the retained target population.','tab:overlap_sensitivity','lrrrrrr',rows,compact=True)
    stair=csv(STAIR/'staircase_metrics.csv')
    wide=stair.pivot(index=['target','n_labeled','support_seed'],columns='stage',values='auroc_eval_group')
    rng=np.random.default_rng(20260922)
    rows=[['Increment','Mean','95\\%seed interval'],r'\midrule']
    for high,low in [('S1','S0'),('S2','S1'),('S3','S2')]:
        blocks=(wide[high]-wide[low]).groupby('support_seed').mean()
        lo,hi=bootstrap_mean(blocks,rng)
        rows.append([high+' - '+low,f'{blocks.mean():+.3f}',f'[{lo:+.3f}, {hi:+.3f}]'])
    table('staircase_intervals.tex','Paired S0--S3 increments with support-seed block bootstrap intervals. Each block preserves all four domains and three budgets; intervals describe seed variation on fixed domains and test sets.','tab:staircase_intervals','lrr',rows)
    cal=pd.concat([csv(SENS/f'calibration_{t}.csv') for t in DOMAINS],ignore_index=True)
    rows=[['Target','Source FPR/TPR','+100 groups FPR/TPR','+500 groups FPR/TPR'],r'\midrule']
    for t in DOMAINS:
        z=cal[(cal.target==t)&(cal.n_labeled==1000)&(cal['mode']=='full_ft')]
        pairs=[]
        a=z[z.n_calibration==100]
        pairs.append(f'{a.source_fpr_row.mean():.3f}/{a.source_tpr_row.mean():.3f}')
        for n in (100,500):
            a=z[z.n_calibration==n];pairs.append(f'{a.fpr_row.mean():.3f}/{a.tpr_row.mean():.3f}')
        rows.append([DISPLAY[t],*pairs])
    table('target_calibration.tex','Row-level operating points after independent target calibration for Full-FT at $N=1000$. Additional calibration groups exclude adaptation support groups. Each cell averages 15 seed coordinates.','tab:target_calibration','lrrr',rows)
    sampling=pd.concat([csv(SENS/f'sampling_{t}.csv') for t in DOMAINS],ignore_index=True)
    rows=[['Target','Balanced','Uniform group','Row arrival'],r'\midrule']
    for t in DOMAINS:
        z=sampling[(sampling.target==t)&(sampling.n_labeled==100)&(sampling.stage=='S3')]
        rows.append([DISPLAY[t],*[f'{z[z.sampling==p].group_auroc.mean():.3f}' for p in ('balanced','uniform_group','row_arrival')]])
    table('passive_sampling.tex','S3 group-level AUROC at $N=100$, averaged over five support seeds. Passive sampling does not use labels to determine arrival order; row-arrival sampling skips groups already selected.','tab:passive_sampling','lrrr',rows)
    neural=csv(SENS/'passive_neural_coordinates.csv')
    rows=[['Target','Mode','Balanced row AUROC','Passive row AUROC','Difference','Passive group AUROC'],r'\midrule']
    for t in DOMAINS:
        for m in ('full_ft','target_only'):
            z=neural[(neural.target==t)&(neural['mode']==m)]
            rows.append([DISPLAY[t],'Full-FT' if m=='full_ft' else 'Target-only',f'{z.balanced_auroc_row.mean():.3f}',
                         f'{z.auroc_row.mean():.3f}',f'{z.delta_auroc_row.mean():+.3f}',f'{z.auroc_group.mean():.3f}'])
    table('passive_neural_rq4.tex','Neural adaptation with passive group sampling at $N=100$. Each cell averages three model seeds and five support seeds. Balanced and passive runs used different GPUs; differences describe protocol sensitivity.','tab:passive_neural_rq4','llrrrr',rows)
    from CODE.analysis.paper_results import DEPLOY
    INPUTS.add(DEPLOY/'summary_rows.csv')
    rows=[['FPR ceiling','TPR$\\ge0.10$','TPR$\\ge0.20$','TPR$\\ge0.30$'],r'\midrule']
    for ceiling in (.01,.05,.10):
        rows.append([f'{ceiling:.2f}',*[int(gates(fpr_ceiling=ceiling,tpr_floor=f).operating_pass.sum()) for f in (.1,.2,.3)]])
    table('gate_sensitivity.tex','Operating-point criterion sensitivity: number of configurations, out of 13, passing on at least three domains. Only evaluation criteria change; models and thresholds remain fixed.','tab:gate_sensitivity','lrrr',rows)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(generator='CODE/analysis/derive_paper_tables.py',generator_sha256=sha(Path(__file__)),
        operation='Archived-result arithmetic only; no figures generated or modified',
        inputs=[dict(path=results.input_path(p),sha256=sha(p)) for p in sorted(INPUTS)],
        outputs=[dict(path=p.relative_to(OUTPUT_ROOT).as_posix(),sha256=sha(p)) for p in sorted(OUTPUTS)])
    (OUTPUT_ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Derived 8 supplementary table summaries and 2 statistical summaries: {OUTPUT_ROOT}')

if __name__=='__main__':main()
