#!/usr/bin/env python3
"""Strict NFv2 S0--S3 staircase using frozen stage3 score caches.

Source models remain frozen throughout S0--S2; S2 fits a target-support logistic
score combiner. S3 is a target-support logistic probe on the loader's
source-fitted 78D representation (the scaler is fit on
source train rows only).  Outputs are compact CSV/JSON; prediction arrays are
never persisted.
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_COMMON_ROOT = _Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_ROOT) not in sys.path: sys.path.insert(0, str(_COMMON_ROOT))
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PAPER = Path(__file__).resolve().parents[2]
if str(PAPER) not in sys.path: sys.path.insert(0, str(PAPER))
from nidsbed.datasets import LODODataLoader, FEW_SHOT_POLICY_GROUP_NESTED, unique_row_weights

METHODS = ("erm", "dann", "coral", "groupdro", "irm", "fishr", "boda")
TARGETS = ("unsw", "cic", "bot", "ton")
SUPPORT_SEEDS = (11, 23, 37, 53, 71)
NS = (20, 50, 100)

def auc(y, s, w=None):
    try: return float(roc_auc_score(y, s, sample_weight=w))
    except ValueError: return float("nan")

def cdf_fit_transform(train, values):
    x = np.sort(np.asarray(train, dtype=np.float64))
    # Right-continuous empirical CDF; no target labels used.
    return np.searchsorted(x, np.asarray(values), side="right") / max(1, len(x))

def cache(score_root, method, target):
    p = score_root / f"{method}_{target}" / "final" / "seed2026_artifacts"
    return {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in
            ("source_val_scores", "source_val_labels", "target_scores", "target_labels")}

def run(root, out, smoke=False, score_root=None):
    score_root = score_root or root / "RESULTS" / "stage3_lodo" / "stage3"
    targets = TARGETS[:1] if smoke else TARGETS
    methods = METHODS[:2] if smoke else METHODS
    ns = (20,) if smoke else NS
    seeds = (11,) if smoke else SUPPORT_SEEDS
    rows=[]; manifest={"protocol":"nfv2_s0_s3_v1","methods":methods,"targets":targets,
                      "support_seeds":seeds,"n_labeled":ns,"cache_seed":2026,
                      "s0":"source-val empirical CDF, equal-weight mean, source AUROC orientation",
                      "s1":"support AUROC orientation of each frozen method score, then equal-weight mean",
                      "s2":"C=.1 LogisticRegression on 7 S1 CDF scores",
                      "s3":"source-train StandardScaler + support LogisticRegression on native 78D"}
    for target in targets:
        loader=LODODataLoader(data_dir=str(root/"DATA"/"nfv2_processed"), target_domain=target,
            target_few_shot_n=0, target_few_shot_balanced=True,
            target_few_shot_policy=FEW_SHOT_POLICY_GROUP_NESTED)
        loader.prepare()
        y=np.asarray(loader._target_y)
        # Duplicate-group equal weights are computed from the unscaled target
        # rows and applied to the fixed target evaluation mask.
        w_full=unique_row_weights(np.asarray(loader._target_X_unscaled))
        caches={m:cache(score_root,m,target) for m in methods}
        src_cdf={}; tgt_cdf={}
        for m,c in caches.items():
            sv=np.asarray(c['source_val_scores']); tv=np.asarray(c['target_scores'])
            orient=1 if auc(c['source_val_labels'],sv)>=0.5 else -1
            src_cdf[m]=orient*cdf_fit_transform(sv,sv)
            tgt_cdf[m]=orient*cdf_fit_transform(sv,tv)
        M=np.column_stack([tgt_cdf[m] for m in methods]); S=np.column_stack([src_cdf[m] for m in methods])
        s0=M.mean(1)
        # Canonical source-fitted 78D LODO representation. The loader fits
        # caps, medians, scaling and invalid indicators on source train rows.
        Xtarget=np.asarray(loader._target_X, dtype=np.float32)
        for seed in seeds:
            for n in ns:
                loader.configure_few_shot_split(n=n, seed=seed, balanced=True, policy=FEW_SHOT_POLICY_GROUP_NESTED)
                # N is the total number of selected labelled duplicate groups.
                # No support-validation tuning is used in this fixed probe; all
                # selected support representatives are available to S1--S3.
                fit=loader._few_shot_idx; ev=loader._cross_eval_mask
                audit=loader.get_few_shot_audit()
                if audit.get("support_eval_group_overlap") != 0:
                    raise RuntimeError(f"support/test group overlap for {target}/N{n}/seed{seed}")
                n_support_groups=int(audit.get("n_support_groups", len(fit)))
                # S1 independently orients each frozen score using support labels.
                # S2 learns unconstrained LR coefficients on these columns.
                # It can reverse their signs; S1 is an interpretable probe,
                # not an additional expressivity restriction on S2.
                score_signs=np.array([1 if auc(y[fit],M[fit,j])>=0.5 else -1 for j in range(M.shape[1])], dtype=np.float64)
                M1=M*score_signs
                s1=M1.mean(1)
                lr=LogisticRegression(C=.1, max_iter=1000, solver="lbfgs").fit(M1[fit], y[fit])
                s2=lr.predict_proba(M1)[:,1]
                lr3=LogisticRegression(C=.1, max_iter=1000, solver="lbfgs").fit(Xtarget[fit], y[fit])
                s3=lr3.predict_proba(Xtarget)[:,1]
                for stage,s in (("S0",s0),("S1",s1),("S2",s2),("S3",s3)):
                    rows.append({"target":target,"n_labeled":n,"support_seed":seed,"stage":stage,
                                 "n_support_rows":int(len(fit)),"n_support_groups":n_support_groups,
                                 "n_eval":int(ev.sum()),
                                 "auroc_eval":auc(y[ev],s[ev]),"auroc_eval_group":auc(y[ev],s[ev],w_full[ev]),
                                 "auroc_all":auc(y,s),"auroc_all_group":auc(y,s,w_full),
                                 "support_auroc":auc(y[fit],s[fit]),"orientation":int(np.sign(score_signs.mean())),
                                 "n_score_flips":int((score_signs<0).sum())})
    out.mkdir(parents=True,exist_ok=True)
    with (out/"staircase_metrics.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    manifest["rows"]=len(rows); manifest["code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return rows

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[2] / "RESULTS/reruns/staircase")
    ap.add_argument("--score-root",type=Path,help="Stage3 output containing method_target/final/seed2026_artifacts")
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args(); run(a.root,a.out,a.smoke,a.score_root)
