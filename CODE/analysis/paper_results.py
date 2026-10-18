"""Paths and coordinate tables used by both the audit and the paper renderer."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FROZEN = ROOT / "RESULTS_PAPER"
RQ1 = FROZEN / "RQ1_source_confidence_transfer/EXP01_stage3_lodo_main_matrix"
RQ2 = FROZEN / "RQ2_failure_heterogeneity"
Q = FROZEN / "RQ3_target_information_recovery/EXP06_qtransport_paired_source_lr/nine_seed"
CONTROL = FROZEN / "RQ3_target_information_recovery/EXP07_matched_native_control_attribution/nine_seed"
STAIR = FROZEN / "RQ3_target_information_recovery/EXP08_s0_s3_staircase"
DEPLOY = FROZEN / "RQ4_ranking_vs_deployment/EXP09_deployment_operating_audit"
SENS = FROZEN / "sensitivity"
DOMAINS = ("unsw", "cic", "bot", "ton")
DISPLAY = {"unsw": "UNSW", "cic": "CIC", "bot": "BoT", "ton": "ToN"}
METHODS = ("erm", "dann", "coral", "groupdro", "irm", "fishr", "boda")
METHOD_DISPLAY = dict(zip(METHODS, ("ERM-cw", "DANN-source", "CORAL", "GroupDRO", "IRM", "Fishr-lite", "BoDA-style")))
SEEDS = (1, 7, 42, 99, 1234, 2026, 2027, 2028, 55555)
SUPPORT_SEEDS = (11, 23, 37, 53, 71)

def configure_results(directory):
    """Select one complete evidence tree, without falling back to archived data."""
    global FROZEN, RQ1, RQ2, Q, CONTROL, STAIR, DEPLOY, SENS
    FROZEN = Path(directory).expanduser().resolve()
    RQ1 = FROZEN / "RQ1_source_confidence_transfer/EXP01_stage3_lodo_main_matrix"
    RQ2 = FROZEN / "RQ2_failure_heterogeneity"
    RQ3 = FROZEN / "RQ3_target_information_recovery"
    Q = RQ3 / "EXP06_qtransport_paired_source_lr/nine_seed"
    CONTROL = RQ3 / "EXP07_matched_native_control_attribution/nine_seed"
    STAIR = RQ3 / "EXP08_s0_s3_staircase"
    DEPLOY = FROZEN / "RQ4_ranking_vs_deployment/EXP09_deployment_operating_audit"
    SENS = FROZEN / "sensitivity"

def input_path(path):
    """Keep in-repository paths portable; allow explicit external input trees."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rq1():
    return pd.read_csv(RQ1 / "Table_3_raw.csv")

def prior_correlations(structure, results):
    """Descriptive correlations; both axes use all retained target records.

    Group priors count label-consistent exact patterns once. Domain-level
    values average the seven method AUROCs before computing the correlation.
    Neither coordinate is an independent-sample significance test.
    """
    import numpy as np
    s = structure.set_index('target').loc[list(DOMAINS)]
    priors = {'row': s.n_attack_rows / s.n_rows,
              'group': s.n_attack_groups / s.n_groups}
    shifts = {name: (4 * p - p.sum()) / 3 for name, p in priors.items()}
    coordinates = {}
    for xname, shift in shifts.items():
        for yname, column in [('row', 'cross_auroc_mean'),
                              ('group', 'cross_unique_auroc_mean')]:
            cell_r = float(np.corrcoef(results.target.map(shift), results[column])[0, 1])
            domain_y = results.groupby('target')[column].mean().reindex(DOMAINS)
            domain_r = float(np.corrcoef(shift, domain_y)[0, 1])
            coordinates[f'x_{xname}__y_{yname}'] = dict(
                cell_r=cell_r, cell_r_squared=cell_r**2, cell_n=len(results),
                domain_r=domain_r, domain_r_squared=domain_r**2, domain_n=len(s))
    return dict(
        population='All retained target records after sampling and preprocessing; no test-split restriction',
        variables=dict(x='Target attack prior minus the equally weighted mean prior of the other three domains',
                       y='Three-seed mean target AUROC for each of seven methods in Table_3_raw.csv'),
        domain_attack_rates_row_level_full_data=priors['row'].to_dict(),
        domain_attack_rates_group_level_full_data=priors['group'].to_dict(),
        domain_signed_prior_shift_row_coordinate=shifts['row'].to_dict(),
        domain_signed_prior_shift_group_coordinate=shifts['group'].to_dict(),
        coordinate_sensitivity=coordinates,
        caveats=['The 28 method-target conditions share only four domain prior values; they are not independent domain samples.',
                 'Correlations and squared correlations are descriptive; they do not establish a causal contribution of prior shift.'])


def tabular_raw(seed_metrics):
    """Rebuild the traditional-model summary from the frozen final seed records."""
    rows = []
    for (method, target), z in seed_metrics.groupby(['method', 'target'], sort=False):
        if z.hp_id.nunique() != 1 or sorted(z.seed) != [2026, 2027, 2028]:
            raise ValueError(f'{method}/{target}: inconsistent final seed records')
        row = dict(method=method, target=target, hp_id=int(z.hp_id.iloc[0]))
        for metric in ('cross_attack_f1', 'cross_auroc', 'cross_auprc',
                       'in_attack_f1', 'in_auroc', 'in_auprc', 'degradation'):
            row[metric+'_mean'] = float(z[metric].mean())
            if metric.startswith('cross_'):
                row[metric+'_std'] = float(z[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def q_coordinates():
    q = pd.DataFrame(read_json(Q / "metrics_long.json"))
    c = pd.DataFrame(read_json(CONTROL / "metrics.json"))
    if "source_seed" in c:
        c = c.rename(columns={"source_seed": "seed"})
    arms = q[q.method.isin(("q_expert_median_inductive", "source_erm"))].copy()
    native = c[c.method == "native78_shared_expert_median"].copy()
    native["method"] = "native78"
    return pd.concat([arms, native], ignore_index=True)

def q_pairs(metric="group_auroc"):
    return q_coordinates().pivot(index=["target", "seed"], columns="method", values=metric)

def staircase():
    return pd.read_csv(STAIR / "staircase_metrics.csv")

def deployment():
    return pd.read_csv(DEPLOY / "summary_rows.csv")

def gates(frame=None, rank_floor=.7, fpr_ceiling=.05, tpr_floor=.2, required_domains=3):
    d = deployment() if frame is None else frame
    records = []
    for (n, mode), cells in d.groupby(["n_labeled", "mode"]):
        row = cells.auroc_row_mean >= rank_floor
        group = cells.auroc_group_mean >= rank_floor
        operating = (cells.source_threshold_fpr_row_mean <= fpr_ceiling) & (cells.source_threshold_tpr_row_mean >= tpr_floor)
        records.append(dict(n_labeled=int(n), mode=mode, row_domains=int(row.sum()),
                            group_domains=int(group.sum()), both_domains=int((row & group).sum()),
                            operating_domains=int(operating.sum()), rank_pass=bool((row & group).all()),
                            operating_pass=bool(operating.sum() >= required_domains)))
    return pd.DataFrame(records).sort_values(["n_labeled", "mode"])

def bootstrap_mean(values, rng, repetitions=10000):
    import numpy as np
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(repetitions, len(v)))].mean(axis=1)
    return np.quantile(means, [.025, .975])
