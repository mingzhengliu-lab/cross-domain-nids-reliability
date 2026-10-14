"""Derive manuscript tables from archived coordinates; preserve all figures.

This performs summary arithmetic only. Concept figures are edited in their SVG
files and exported with Inkscape; empirical figures remain the original assets.
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
    PAPER, RQ1, RQ2, SENS, STAIR, DOMAINS, DISPLAY,
    rq1, staircase, gates, bootstrap_mean, prior_correlations, tabular_raw,
)
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
    p = PAPER/'tables'/name
    p.write_text('% Derived by CODE/analysis/derive_paper_tables.py from archived results.\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    OUTPUTS.append(p)

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for d in ('tables','generated'): (PAPER/d).mkdir(parents=True,exist_ok=True)
    d=csv(RQ1/'Table_3_raw.csv'); cache=csv(SENS/'rq1_cache_metrics.csv')
    prior = prior_correlations(csv(SENS/'data_structure.csv'), d)
    prior['inputs'] = [dict(path=p.relative_to(ROOT).as_posix(),
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in (SENS/'data_structure.csv', RQ1/'Table_3_raw.csv')]
    path = RQ2/'EXP05_domain_fingerprint_priorshift/prior_shift_correlation.json'
    path.write_text(json.dumps(prior,indent=2)+'\n',encoding='utf-8')
    OUTPUTS.append(path)
    traditional = RQ2/'EXP03_model_family_controls/traditional_supervised'
    raw_path = traditional/'tabular_raw.csv'
    tabular_raw(csv(traditional/'tabular_seed_metrics.csv')).to_csv(raw_path,index=False)
    OUTPUTS.append(raw_path)
    rows=[['目标','源行AUROC','目标行AUROC','源组AUROC','目标组AUROC','目标行AUPRC'],r'\midrule']
    for t in DOMAINS:
        z=d[d.target==t];c=cache[cache.target==t]
        rows.append([DISPLAY[t],f'{z.in_auroc_mean.mean():.4f}',f'{z.cross_auroc_mean.mean():.4f}',
                     f'{c.source_group_auroc.mean():.4f}',f'{z.cross_unique_auroc_mean.mean():.4f}',f'{z.cross_auprc_mean.mean():.4f}'])
    table('rq1_matched.tex','同口径的源内与目标结果。行级沿用冻结RQ1记录；源组级从归档源内部测试分数及权重复算。','tab:rq1_matched','lrrrrr',rows)
    family=csv(RQ2/'EXP03_model_family_controls/comparison/baseline_results.csv')
    rows=[['方法','UNSW组级','CIC组级','BoT组级','ToN组级','行级宏均值','组级宏均值'],r'\midrule']
    family_names={'logreg':'LogReg','histgb':'HistGB','extratrees':'ExtraTrees','lightgbm':'LightGBM',
                  'mahalanobis':'Mahalanobis','iforest':'IsolationForest','autoencoder':'Autoencoder','deep_svdd':'DeepSVDD'}
    for category,title in [('traditional','传统源监督'),('anomaly','源正常样本拟合的异常检测')]:
        rows.append(r'\multicolumn{7}{l}{\textit{'+title+r'}} \\')
        for method in family.loc[family.category==category,'method'].drop_duplicates():
            z=family[(family.category==category)&(family.method==method)].set_index('target').loc[list(DOMAINS)]
            rows.append([family_names[method],*[f'{v:.4f}' for v in z.group_auroc_mean],
                         f'{z.row_auroc_mean.mean():.4f}',f'{z.group_auroc_mean.mean():.4f}'])
        if category=='traditional': rows.append(r'\midrule')
    table('cross_family_baselines.tex','模型家族对照在保留目标全集上的AUROC，由归档预测分数复算。各域数值先对三个训练种子（2026、2027、2028）取均值，宏均值再对四域等权平均。HistGB为HistGradientBoosting；两类模型的源信息条件分别列示。','tab:cross_family_baselines','lrrrrrr',rows,compact=True)
    overlap=csv(SENS/'exact_overlap.csv'); sensitivity=csv(SENS/'overlap_sensitivity.csv')
    rows=[['目标','测试行重合','测试行异标签','测试组异标签','全域行重合','剔除后行AUC','剔除后组AUC'],r'\midrule']
    for t in DOMAINS:
        z=overlap[overlap.target==t].iloc[0];s=sensitivity[sensitivity.target==t]
        rows.append([DISPLAY[t],f'{100*z.test_row_any_match:.3f}',f'{100*z.test_row_opposite_label:.3f}',
                     f'{100*z.test_group_opposite_label:.3f}',f'{100*z.full_row_any_match:.3f}',
                     f'{s.excluded_row_auroc.mean():.4f}',f'{s.excluded_group_auroc.mean():.4f}'])
    table('overlap_sensitivity.tex','源训练分区与目标精确特征重合及RQ1剔除敏感性。重合与异标签比例以百分数表示；最后两列在保留目标全集中剔除全部源训练重合组后复算。','tab:overlap_sensitivity','lrrrrrr',rows,compact=True)
    stair=csv(STAIR/'staircase_metrics.csv')
    wide=stair.pivot(index=['target','n_labeled','support_seed'],columns='stage',values='auroc_eval_group')
    rng=np.random.default_rng(20260922)
    rows=[['配对增量','均值','95\\%种子区间'],r'\midrule']
    for high,low in [('S1','S0'),('S2','S1'),('S3','S2')]:
        blocks=(wide[high]-wide[low]).groupby('support_seed').mean()
        lo,hi=bootstrap_mean(blocks,rng)
        rows.append([high+'减'+low,f'{blocks.mean():+.3f}',f'[{lo:+.3f}, {hi:+.3f}]'])
    table('staircase_intervals.tex','S0--S3配对增量及支持种子块bootstrap区间。每个种子块保留四域和三个预算的全部配对；区间只描述固定域和测试集上的种子变异。','tab:staircase_intervals','lrr',rows)
    cal=pd.concat([csv(SENS/f'calibration_{t}.csv') for t in DOMAINS],ignore_index=True)
    rows=[['目标','源阈值FPR/TPR','加100组FPR/TPR','加500组FPR/TPR'],r'\midrule']
    for t in DOMAINS:
        z=cal[(cal.target==t)&(cal.n_labeled==1000)&(cal['mode']=='full_ft')]
        pairs=[]
        a=z[z.n_calibration==100]
        pairs.append(f'{a.source_fpr_row.mean():.3f}/{a.source_tpr_row.mean():.3f}')
        for n in (100,500):
            a=z[z.n_calibration==n];pairs.append(f'{a.fpr_row.mean():.3f}/{a.tpr_row.mean():.3f}')
        rows.append([DISPLAY[t],*pairs])
    table('target_calibration.tex','$N=1000$、Full-FT下独立目标校准的行级均值运行点。额外标签预算不包含适应支持组，每格汇总15个种子坐标。','tab:target_calibration','lrrr',rows)
    sampling=pd.concat([csv(SENS/f'sampling_{t}.csv') for t in DOMAINS],ignore_index=True)
    rows=[['目标','每类平衡','组均匀被动','行到达被动'],r'\midrule']
    for t in DOMAINS:
        z=sampling[(sampling.target==t)&(sampling.n_labeled==100)&(sampling.stage=='S3')]
        rows.append([DISPLAY[t],*[f'{z[z.sampling==p].group_auroc.mean():.3f}' for p in ('balanced','uniform_group','row_arrival')]])
    table('passive_sampling.tex','$N=100$时S3的组级AUROC（五个支持种子均值）。被动策略不按标签决定采集顺序；行到达抽样跳过已选重复组。','tab:passive_sampling','lrrr',rows)
    neural=csv(SENS/'passive_neural_coordinates.csv')
    rows=[['目标','模式','平衡行AUROC','被动行AUROC','差值','被动组AUROC'],r'\midrule']
    for t in DOMAINS:
        for m in ('full_ft','target_only'):
            z=neural[(neural.target==t)&(neural['mode']==m)]
            rows.append([DISPLAY[t],'Full-FT' if m=='full_ft' else 'Target-only',f'{z.balanced_auroc_row.mean():.3f}',
                         f'{z.auroc_row.mean():.3f}',f'{z.delta_auroc_row.mean():+.3f}',f'{z.auroc_group.mean():.3f}'])
    table('passive_neural_rq4.tex','神经适应在$N=100$下的被动组抽样敏感性。每格汇总三个模型种子与五个支持种子。平衡与被动实验使用不同GPU，差值作为协议敏感性描述。','tab:passive_neural_rq4','llrrrr',rows)
    from CODE.analysis.paper_results import DEPLOY
    INPUTS.add(DEPLOY/'summary_rows.csv')
    rows=[['FPR上限','TPR$\\ge0.10$','TPR$\\ge0.20$','TPR$\\ge0.30$'],r'\midrule']
    for ceiling in (.01,.05,.10):
        rows.append([f'{ceiling:.2f}',*[int(gates(fpr_ceiling=ceiling,tpr_floor=f).operating_pass.sum()) for f in (.1,.2,.3)]])
    table('gate_sensitivity.tex','运行点判据的事后敏感性：13个配置中至少三域均值达标的配置数。只改变评价判据，不重新选择模型或阈值。','tab:gate_sensitivity','lrrr',rows)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(generator='CODE/analysis/derive_paper_tables.py',generator_sha256=sha(Path(__file__)),
        operation='Archived-result arithmetic only; no figures generated or modified',
        inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(INPUTS)],
        outputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(OUTPUTS)])
    (PAPER/'generated/artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Derived 8 manuscript tables and 2 statistical summaries from {len(INPUTS)} archived inputs; figures unchanged.')

if __name__=='__main__':main()
