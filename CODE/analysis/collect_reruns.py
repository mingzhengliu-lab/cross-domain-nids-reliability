"""Assemble a separate paper-results tree from fresh experiment outputs only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from CODE.analysis.output_paths import output_directory
from CODE.analysis import paper_results as results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reruns-root', type=Path, default=ROOT/'RESULTS/reruns')
    p.add_argument('--output-dir', type=Path, default=ROOT/'RESULTS/reruns/paper_results')
    args = p.parse_args()
    run = args.reruns_root.resolve()
    out = output_directory(args.output_dir)
    results.configure_results(out)
    pairs = []

    def add(source, dest):
        pairs.append((Path(source), Path(dest)))

    add(run/'stage3_summary/Table_3_raw.csv', results.RQ1/'Table_3_raw.csv')
    for method in results.METHODS:
        for target in results.DOMAINS:
            cell = f'{method}_{target}'
            for name in ['summary.json']+[f'seed{s}.json' for s in (2026, 2027, 2028)]:
                add(run/'stage3'/cell/'final'/name, results.RQ1/'per_cell'/cell/name)
    family = results.RQ2/'EXP03_model_family_controls'
    add(run/'families_summary/baseline_results.csv', family/'comparison/baseline_results.csv')
    add(run/'families_summary/tabular_seed_metrics.csv', family/'traditional_supervised/tabular_seed_metrics.csv')
    add(run/'tabular_aggregate/tabular_raw.csv', family/'traditional_supervised/tabular_raw.csv')
    diagnostics = results.RQ2/'EXP05_domain_fingerprint_priorshift'
    for name in ('summary.json', 'domain_rows.csv', 'domain_summary.csv', 'model_rows.csv', 'model_summary.csv', 'mechanism_relations.csv'):
        add(run/'domain_fingerprint/aggregate'/name, diagnostics/name)
    for name in ('summary.json', 'coordinate_rows.csv', 'summary_rows.csv', 'target_audit_rows.csv'):
        add(run/'label_conflict/aggregate'/name, diagnostics/'conditional_label_shift'/name)
    add(run/'qtransport/metrics_long.json', results.Q/'metrics_long.json')
    add(run/'matched_control/metrics.json', results.CONTROL/'metrics.json')
    add(run/'staircase/staircase_metrics.csv', results.STAIR/'staircase_metrics.csv')
    for name in ('staircase_macro.csv', 'staircase_by_target_budget.csv'):
        add(run/'staircase_summary'/name, results.STAIR/name)
    for name in ('coordinate_rows.csv', 'summary_rows.csv', 'paired_deltas.csv', 'budget_gates.json'):
        add(run/'deployment_summary'/name, results.DEPLOY/name)
    for name in ('data_structure.csv', 'exact_overlap.csv', 'overlap_sensitivity.csv', 'rq1_cache_metrics.csv', 'source_internal_overlap.csv', 'source_internal_sensitivity.csv'):
        add(run/'data_checks'/name, results.SENS/name)
    for target in results.DOMAINS:
        add(run/'support_controls'/f'sampling_{target}.csv', results.SENS/f'sampling_{target}.csv')
        add(run/'calibration'/f'calibration_{target}.csv', results.SENS/f'calibration_{target}.csv')
    for name in ('ton_split_sensitivity.csv', 'ton_alternative_splits.csv'):
        add(run/'support_controls'/name, results.SENS/name)
    add(run/'passive_summary/passive_neural_coordinates.csv', results.SENS/'passive_neural_coordinates.csv')
    missing = [str(source) for source, _ in pairs if not source.is_file()]
    if missing:
        raise FileNotFoundError('Complete the preceding experiment/summary steps first:\n'+'\n'.join(missing))
    if any(source.resolve() == dest.resolve() for source, dest in pairs):
        raise ValueError('Collected outputs must be separate from their inputs')
    manifest = []
    for source, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest.append(dict(source=str(source.relative_to(run)), output=str(dest.relative_to(out)),
                             sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (out/'collection_manifest.json').write_text(json.dumps({'inputs_root': str(run), 'files': manifest}, indent=2)+'\n', encoding='utf-8')
    print(f'Collected {len(pairs)} newly generated files into {out}')


if __name__ == '__main__':
    main()
