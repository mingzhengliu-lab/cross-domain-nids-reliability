"""Check manuscript values against frozen evidence; write reports under RESULTS.
No training, table generation into paper/, or mutation of archived evidence.
"""
from pathlib import Path
import sys, json, re, hashlib, itertools, argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'RESULTS/reruns/manuscript_checks'
sys.path.insert(0, str(ROOT))
from CODE.analysis import paper_results as p

ap = argparse.ArgumentParser()
ap.add_argument('--project', type=Path, default=ROOT/'paper')
ap.add_argument('--tag', default='current')
ap.add_argument('--output-dir', type=Path, default=OUT)
args=ap.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
tex=(args.project/'main_v1.tex').read_text(encoding='utf-8-sig')
checks=[]; comparisons=[]; inputs=set()
def read(path):
    inputs.add(Path(path)); return pd.read_csv(path)
def check(name, condition, details=None):
    checks.append(dict(id=f'EN-{len(checks)+1:03}', name=name, passed=bool(condition), details=details))
def table(label):
    return tex.split('\\label{'+label+'}',1)[1].split('\\end{table}',1)[0]
def rowcheck(label, name, expected, digits=3, drop=1):
    block=table(label)
    match=re.search(r'^\s*'+re.escape(name)+r'\s*&\s*(.+)',block,re.M)
    tokens=[]
    if match:
        for cell in match[0].split('&')[drop:]:
            cell=re.sub(r'\\multicolumn.*','',cell)
            cell=cell.replace(',', '')
            tokens.extend(re.findall(r'[-+]?\d+\.?\d*',cell))
    actual=[float(v) for v in tokens]
    rounded=[float(f'{v:.{digits}f}') for v in expected]
    ok=len(actual)==len(rounded) and actual==rounded
    comparisons.append(dict(table=label,row=name,original_values=actual,
                            unrounded_evidence=[float(v) for v in expected],
                            expected_display=rounded,passed=ok))
    check(f'{label}: {name}',ok, comparisons[-1] if not ok else None)

d=read(p.RQ1/'Table_3_raw.csv')
cache=read(p.SENS/'rq1_cache_metrics.csv')
structure=read(p.SENS/'data_structure.csv').set_index('target')
for t in p.DOMAINS:
    s=structure.loc[t]
    rowcheck('tab:dataset_statistics',p.DISPLAY[t],
             [s.n_rows,s.n_groups,s.test_rows,s.test_groups],0)
    z=d[d.target==t]; c=cache[cache.target==t]
    rowcheck('tab:target_performance',p.DISPLAY[t],
             [z.in_auroc_mean.mean(),c.source_group_auroc.mean(),z.cross_auroc_mean.mean(),
              z.cross_unique_auroc_mean.mean(),int((z.cross_auroc_mean<.5).sum()),7,
              int((z.cross_unique_auroc_mean<.5).sum()),7],4)
check('28 three-seed RQ1 conditions',len(d)==28 and len(cache)==84)
headline=dict(source_mean=d.in_auroc_mean.mean(),source_sd=d.in_auroc_mean.std(),
              target_row_mean=d.cross_auroc_mean.mean(),target_row_sd=d.cross_auroc_mean.std(),
              target_group_mean=d.cross_unique_auroc_mean.mean(),target_group_sd=d.cross_unique_auroc_mean.std(),
              gap=(d.in_auroc_mean-d.cross_auroc_mean).mean(),
              row_below=int((d.cross_auroc_mean<.5).sum()),group_below=int((d.cross_unique_auroc_mean<.5).sum()))
for token,v in [('0.9465',headline['source_mean']),('0.0400',headline['source_sd']),
                ('0.3349',headline['target_row_mean']),('0.1607',headline['target_row_sd']),
                ('0.3846',headline['target_group_mean']),('0.1409',headline['target_group_sd']),
                ('0.6116',headline['gap'])]:
    check('RQ1 headline '+token, f'{v:.4f}'==token and token in tex)
delta=d.cross_unique_auroc_mean-d.cross_auroc_mean
for token,v in [('0.0498',delta.mean()),('0.0523',delta.median()),('-0.1085',delta.min()),('0.1479',delta.max())]:
    check('duplicate-weighting statistic '+token,f'{v:.4f}'==token)
family=read(p.RQ2/'EXP03_model_family_controls/comparison/baseline_results.csv')
fm={'LogReg':'logreg','HistGB':'histgb','ExtraTrees':'extratrees','LightGBM':'lightgbm',
    'Mahalanobis':'mahalanobis','IsolationForest':'iforest','Autoencoder':'autoencoder','DeepSVDD':'deep_svdd'}
for name,m in fm.items():
    z=family[family.method==m].set_index('target').loc[list(p.DOMAINS)]
    rowcheck('tab:model_family_comparison',name,
             [*z.group_auroc_mean,z.row_auroc_mean.mean(),z.group_auroc_mean.mean()],4)
prior=p.prior_correlations(structure.reset_index(),d)
for typ,name in [('row','Row prior / row AUROC'),('group','Group prior / group AUROC')]:
    v=prior['coordinate_sensitivity'][f'x_{typ}__y_{typ}']
    rowcheck('tab:prior_correlation',name,[v['cell_r'],v['cell_r_squared'],28,4],3)
domain=read(p.RQ2/'EXP05_domain_fingerprint_priorshift/domain_rows.csv')
domain=domain.query("representation=='raw39' and task=='all' and model=='logreg'")
ind=read(p.RQ2/'EXP05_domain_fingerprint_priorshift/model_rows.csv').query("representation=='raw39'")
for t in p.DOMAINS:
    z=ind[ind.outer_target==t]
    rowcheck('tab:domain_fingerprint',p.DISPLAY[t],
             [domain[domain.outer_target==t].macro_ovr_auroc.mean(),z.row_auroc.mean(),z.group_auroc.mean(),
              int((z.group_auroc<.5).sum()),3],3)
overlap=read(p.SENS/'exact_overlap.csv'); sensitivity=read(p.SENS/'overlap_sensitivity.csv')
for t in p.DOMAINS:
    z=overlap[overlap.target==t].iloc[0];s=sensitivity[sensitivity.target==t]
    values=[100*z.test_row_any_match,100*z.test_row_opposite_label,100*z.test_group_opposite_label,
            100*z.full_row_any_match,s.excluded_row_auroc.mean(),s.excluded_group_auroc.mean()]
    expected=[float(f'{v:.{3 if i<4 else 4}f}') for i,v in enumerate(values)]
    rowcheck('tab:feature_overlap',p.DISPLAY[t],expected,4)
check('filtered global AUROC',f'{sensitivity.excluded_row_auroc.mean():.4f}'=='0.3200' and
      f'{sensitivity.excluded_group_auroc.mean():.4f}'=='0.3843')
inputs.update([p.Q/'metrics_long.json',p.CONTROL/'metrics.json'])
qp=p.q_pairs()
qsummary=[]
for t in p.DOMAINS:
    z=qp.loc[t];q=z.q_expert_median_inductive;lr=z.source_erm;nat=z.native78
    dl=q-lr;dn=q-nat
    rowcheck('tab:q_transport_results',p.DISPLAY[t],
             [q.mean(),q.std(),lr.mean(),lr.std(),dl.mean(),int((dl>0).sum()),9])
    rowcheck('tab:matched_experts',p.DISPLAY[t],
             [q.mean(),q.std(),nat.mean(),nat.std(),dn.mean(),dn.std(),int((dn>0).sum()),9])
    qsummary.append(dict(target=t,q_mean=q.mean(),q_sd=q.std(),lr_mean=lr.mean(),native_mean=nat.mean(),
                         delta_lr=dl.mean(),delta_native=dn.mean(),native_sd=nat.std()))
stair=read(p.STAIR/'staircase_metrics.csv')
sw=stair.pivot(index=['target','n_labeled','support_seed'],columns='stage',values='auroc_eval_group')
previous=None;stage_summary=[]
for stage in ['S0','S1','S2','S3']:
    z=stair[stair.stage==stage]
    values=z.auroc_eval_group if stage!='S0' else z.drop_duplicates('target').auroc_eval_group
    expected=[values.mean(),values.std()]
    if previous is not None: expected.append((sw[stage]-sw[previous]).mean())
    expected.append(z.support_auroc.mean())
    rowcheck('tab:s0_s3_overall',stage,expected)
    stage_summary.append(dict(stage=stage,mean=values.mean(),sd=values.std(),support_mean=z.support_auroc.mean(),
                              unique_summary_units=len(values)))
    previous=stage
check('S0 four unique baselines',stair[stair.stage=='S0'].groupby('target').auroc_eval_group.nunique().eq(1).all())
rng=np.random.default_rng(20260922)
intervals=[]
for high,low in [('S1','S0'),('S2','S1'),('S3','S2')]:
    blocks=(sw[high]-sw[low]).groupby('support_seed').mean()
    lo,hi=p.bootstrap_mean(blocks,rng)
    rowcheck('tab:stage_bootstrap',high+' $-$ '+low,[blocks.mean(),lo,hi])
    intervals.append(dict(comparison=high+'-'+low,mean=blocks.mean(),lower=lo,upper=hi,blocks=blocks.to_dict()))
by=stair.groupby(['target','n_labeled','stage']).auroc_eval_group.mean().unstack()
by.to_csv(OUT/f'{args.tag}_staircase_domain_budget.csv')
dep=read(p.DEPLOY/'summary_rows.csv'); raw=read(p.DEPLOY/'coordinate_rows.csv')
g=p.gates(dep)
for r in g.itertuples():
    name=f'{r.n_labeled}'+r'\s*&\s*'+{'zero_shot':'Zero-shot','full_ft':'Full-FT','target_only':'Target-only'}[r.mode]
    name=name.replace('\\s*','')
    # Match whitespace-insensitive first two columns separately.
    block=table('tab:ranking_operating_gates')
    mode={'zero_shot':'Zero-shot','full_ft':'Full-FT','target_only':'Target-only'}[r.mode]
    match=re.search(rf'^\s*{r.n_labeled}\s*&\s*{re.escape(mode)}\s*&\s*(.+)',block,re.M)
    actual=[int(x) for x in re.findall(r'\d+',match[1])] if match else []
    expected=[r.row_domains,4,r.group_domains,4,r.both_domains,4,r.operating_domains,4]
    check(f'gate N={r.n_labeled} {mode}',actual==expected and 'No' in match[1])
check('732 records, 52 cells, 13 configurations, five ranking / zero operating passes',
      len(raw)==732 and len(dep)==52 and len(g)==13 and g.rank_pass.sum()==5 and g.operating_pass.sum()==0)
keys=['n_labeled','mode','checkpoint_seed','support_seed']
positive=raw[raw.n_labeled>0].assign(pass_op=lambda x:(x.source_threshold_fpr_row<=.05)&(x.source_threshold_tpr_row>=.2))
individual=positive.groupby(keys).pass_op.sum()
check('five individual cross-domain combinations',int((individual>=3).sum())==5)
examples=[('unsw',1000,'full_ft'),('unsw',1000,'target_only'),('ton',1000,'target_only'),
          ('unsw',0,'zero_shot'),('cic',0,'zero_shot'),('ton',0,'zero_shot'),('bot',10,'full_ft')]
example_results=[]
for t,n,m in examples:
    z=dep[(dep.target==t)&(dep.n_labeled==n)&(dep['mode']==m)].iloc[0]
    mode={'zero_shot':'Zero-shot','full_ft':'Full-FT','target_only':'Target-only'}[m]
    block=table('tab:threshold_examples')
    match=re.search(rf'^\s*{p.DISPLAY[t]}\s*&\s*{n}\s*&\s*{re.escape(mode)}\s*&\s*(.+)',block,re.M)
    actual=[float(x) for x in re.findall(r'\d+\.?\d*',match[1].replace(',',''))]
    expected=[float(f'{z.auroc_row_mean:.3f}'),float(f'{z.source_threshold_fpr_row_mean:.{4 if n==1000 else 3}f}'),
              float(f'{z.source_threshold_tpr_row_mean:.3f}'),round(z.alerts_per_million_row_mean)]
    check(f'operating example {t}/{n}/{m}',actual==expected,dict(actual=actual,expected=expected) if actual!=expected else None)
    example_results.append(dict(target=t,n=n,mode=m,actual=actual,expected=expected))
pi=raw.target.map(structure.test_attack_rate)
recalc_alerts=1e6*(pi*raw.source_threshold_tpr_row+(1-pi)*raw.source_threshold_fpr_row)
check('total alerts use one million all flows',np.allclose(recalc_alerts,raw.alerts_per_million_row,atol=.002,rtol=0))
legacy_false=1e6*raw.source_threshold_fpr_row
check('legacy false-alert field uses one million benign flows',np.allclose(legacy_false,raw.false_alerts_per_million_row,atol=.002,rtol=0))
raw.assign(corrected_false_alerts_per_million_all_flows=1e6*(1-pi)*raw.source_threshold_fpr_row,
           recalculated_total_alerts_per_million_all_flows=recalc_alerts).to_csv(OUT/f'{args.tag}_alert_arithmetic.csv',index=False)
sampling=pd.concat([read(p.SENS/f'sampling_{t}.csv') for t in p.DOMAINS],ignore_index=True)
for t in p.DOMAINS:
    z=sampling[(sampling.target==t)&(sampling.n_labeled==100)&(sampling.stage=='S3')]
    rowcheck('tab:passive_s3',p.DISPLAY[t],[z[z.sampling==s].group_auroc.mean() for s in ['balanced','uniform_group','row_arrival']])
one=sampling[sampling.one_class_support]
check('720 sampling stage records, five single-class support conditions',len(sampling)==720 and len(one)==20 and
      len(one[['target','sampling','n_labeled','support_seed']].drop_duplicates())==5)
neural=read(p.SENS/'passive_neural_coordinates.csv')
for t in p.DOMAINS:
    for m in ['full_ft','target_only']:
        z=neural[(neural.target==t)&(neural['mode']==m)]
        mode='Full-FT' if m=='full_ft' else 'Target-only'
        block=table('tab:passive_neural')
        match=re.search(rf'^\s*{p.DISPLAY[t]}\s*&\s*{mode}\s*&\s*(.+)',block,re.M)
        actual=[float(x) for x in re.findall(r'[-+]?\d+\.\d+',match[1])]
        expected=[float(f'{v:.3f}') for v in [z.balanced_auroc_row.mean(),z.auroc_row.mean(),z.delta_auroc_row.mean(),z.auroc_group.mean()]]
        check(f'passive neural {t}/{m}',actual==expected)
alt=read(p.SENS/'ton_split_sensitivity.csv')
cal=pd.concat([read(p.SENS/f'calibration_{t}.csv') for t in p.DOMAINS],ignore_index=True)
for t in p.DOMAINS:
    z=cal[(cal.target==t)&(cal.n_labeled==1000)&(cal['mode']=='full_ft')]
    values=[z[z.n_calibration==100].source_fpr_row.mean(),z[z.n_calibration==100].source_tpr_row.mean()]
    for n in [100,500]:
        a=z[z.n_calibration==n];values += [a.fpr_row.mean(),a.tpr_row.mean()]
    rowcheck('tab:target_calibration_details',p.DISPLAY[t],values)
check('960 independent calibration conditions, 55 percent underresolved',len(cal)==960 and
      cal.calibration_support_group_overlap.eq(0).all() and cal.calibration_test_group_overlap.eq(0).all() and
      cal.calibration_underresolved.mean()==.55)
for fpr in [.01,.05,.10]:
    rowcheck('tab:gate_sensitivity',f'{fpr:.2f}',
             [p.gates(dep,fpr_ceiling=fpr,tpr_floor=tpr).operating_pass.sum() for tpr in [.1,.2,.3]],0)
source=read(p.SENS/'source_internal_sensitivity.csv')
for col,expected in [('source_row_auroc','0.94654'),('excluded_source_row_auroc','0.96986'),
                     ('source_group_auroc','0.97034'),('excluded_source_group_auroc','0.97041')]:
    check('source exclusion '+col,f'{source[col].mean():.5f}'==expected)
# Remove LaTeX comments before structural checks; escaped percent signs survive.
clean=re.sub(r'(?<!\\)%[^\n]*','',tex)
labels=re.findall(r'\\label\{([^}]+)\}',clean)
refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',clean)
cites={k.strip() for b in re.findall(r'\\cite\w*\{([^}]+)\}',clean) for k in b.split(',')}
bib=(args.project/'cas-refs.bib').read_text(encoding='utf-8-sig')
entries=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib)
figs=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',clean)
check('unique labels and all refs resolve',len(labels)==len(set(labels)) and set(refs)<=set(labels))
check('unique BibTeX keys and all cites resolve',len(entries)==len(set(entries)) and cites<=set(entries),
      dict(cited=len(cites),entries=len(entries),missing=sorted(cites-set(entries))))
check('all nine actual included figures exist',len(figs)==9 and all((args.project/f).is_file() for f in figs))
result=dict(project=str(args.project),scope='Independent English inline-table arithmetic plus structural checks; no retraining',
            passed=sum(c['passed'] for c in checks),total=len(checks),checks=checks,
            headline=headline,q_summary=qsummary,stage_summary=stage_summary,bootstrap=intervals,
            alternative_split_columns=list(alt.columns),
            input_sha256=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(inputs)])
(OUT/f'{args.tag}_independent_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:x.item())+'\n',encoding='utf-8')
(OUT/f'{args.tag}_table_comparisons.json').write_text(json.dumps(comparisons,indent=2,default=lambda x:x.item())+'\n',encoding='utf-8')
print(f"{result['passed']}/{result['total']} English checks passed")
for c in checks:
    if not c['passed']:print(json.dumps(c,ensure_ascii=False))
print('Stage summaries:',stage_summary)

raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
