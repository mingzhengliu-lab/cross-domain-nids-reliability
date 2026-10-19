# Reliability of Cross-Domain Network Intrusion Detection

Research code and archived evidence for **Reliability of Cross-Domain Network
Intrusion Detection**, by MingZheng Liu, Yiyang Fu, Saru Kumari, and Chien-Ming Chen.

[Repository](https://github.com/mingzhengliu-lab/cross-domain-nids-reliability)

## What this study evaluates

Experiments hold out one of four NFv2 domains—UNSW, CIC, BoT, and ToN—and use the
other three as sources. The study separates target ranking, score direction,
and threshold-based detection. It compares what unlabeled target features and
limited target labels can repair, then tests operating points after neural adaptation.

- Across 28 source-trained method–target conditions, mean source row AUROC is
  **0.9465**, target row AUROC **0.3349**, and target group AUROC **0.3846**.
  Twenty target group-level means fall below 0.5.
- Q-TRANSPORT improves ranking on three targets and produces negative transfer
  on ToN. Under matched support labels, S0–S3 attain group-level macro-average
  AUROCs of **0.368, 0.660, 0.689, and 0.906**.
- In a separate neural experiment, **5/13** budget–training-mode configurations
  pass the ranking criterion, while **0/13** pass the source-selected operating-point
  criterion. These counts concern seed-averaged configurations, not every individual run.

RQ1 and model-family controls evaluate all retained target records. Q-TRANSPORT,
S0–S3, and neural adaptation use fixed target test partitions. Comparisons stay
within their respective protocols; these branches are not a common algorithm leaderboard.

## Start with the archived results

The compact checks require no raw traffic, model checkpoints, GPU, or PyTorch.
From the repository root, use Python 3.10 or 3.11:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python CODE/run_paper.py audit
```

`audit` verifies the archived coordinate grids, paired seeds, aggregation,
split-isolation records, operating criteria, and evidence hashes. Reports are
written to `RESULTS/reruns/submission_checks/`. It requires only the distributed
code and archived results; it does not retrain the models.

The manuscript, bibliography, template, and figure assets in the local `paper/`
directory are not distributed with this repository. If those files are available
locally, `python CODE/run_paper.py audit --with-manuscript` additionally checks
their consistency and hashes. `python CODE/run_paper.py check-manuscript --project
/path/to/paper` checks numerical table entries and headline statistics against
`main.tex`, with reports under `RESULTS/reruns/manuscript_checks/`.

## Repository map

| Location | Contents |
|---|---|
| `CODE/` | Experiment runners, preprocessing, aggregation, plotting, and verification |
| `RESULTS_PAPER/` | Compact frozen metrics, per-seed records, and provenance |
| `DATA/nfv2_processed/` | Feature definitions, row indices, duplicate-group keys, and fixed splits |
| `CODE/figures/` | Numerical figure generation from the archived experiment records |

`RESULTS/` and `paper/` are local directories excluded from Git. Raw traffic, model
checkpoints, virtual environments, and local verification outputs are not part
of the release. The fixed row indices, duplicate-group identifiers, and split
files in `DATA/` are retained to reconstruct the experiment populations.
`DATA/nfv2_processed/splits/sensitivity_indices.npz` additionally contains the
fixed support/calibration indices and five alternative ToN partitions. It
contains no traffic features, labels, model weights, or prediction scores.

| Research question | Code | Evidence under `RESULTS_PAPER/` |
|---|---|---|
| RQ1: source-to-target transfer | `CODE/experiment_stage3/` | `RQ1_source_confidence_transfer/` |
| RQ2: failure diagnostics | `CODE/experiment_diagnostics/` | `RQ2_failure_heterogeneity/`, `sensitivity/` |
| RQ3: unlabeled Q-TRANSPORT | `CODE/experiment_low_label/` | `RQ3_target_information_recovery/` |
| RQ3: matched-support S0–S3 | `CODE/experiment_staircase/` | `RQ3_target_information_recovery/` |
| RQ4: adaptation and operating points | `CODE/experiment_low_label/` | `RQ4_ranking_vs_deployment/` |

Evidence directories follow the four research questions in the manuscript.
RQ3 contains Q-TRANSPORT, matched controls, and the S0-S3 staircase; RQ4 contains
neural adaptation and operating-point evaluation. Experiment identifiers
(`EXPxx`) remain stable within those directories.

The staircase summary `staircase_macro.csv` is derived from the per-condition
`staircase_metrics.csv`. Test AUROC means and sample SDs (`ddof=1`) use four
distinct target baselines for S0 and 60 target–budget–support-seed conditions
for each of S1–S3, as recorded in `n_summary_units`. The `n_cells` column counts
all 60 coordinate records per stage. `support_mean` averages their support-set
evaluations; `flip_mean` averages the S1 orientation diagnostic recorded for
those same conditions.

Paired staircase intervals use five support-seed blocks (11, 23, 37, 53, 71).
Each block averages the 12 paired group-AUROC differences across four targets
and three budgets (20, 50, 100). For each comparison, draw five blocks with
replacement 10,000 times and average the selected block means. The 95% interval
uses the 2.5th and 97.5th percentiles (`numpy.quantile`, default linear method).
Initialize `numpy.random.default_rng(20260922)` once and use that same generator
consecutively for S1−S0, S2−S1, and S3−S2, without resetting it between comparisons.
The implementation is in `CODE/analysis/paper_results.py` (`bootstrap_mean`),
with the comparison order in `CODE/analysis/derive_paper_tables.py`.

## Recreate numerical figures and summary tables

With the analysis environment above, run:

```bash
python CODE/run_paper.py figures
python CODE/run_paper.py tables
```

`figures` renders Figures 4-9 as PDF and editable SVG files under
`RESULTS/reruns/figures/`. Each figure also has a CSV containing its plotted
values; a JSON manifest records the input paths, hashes, and aggregation.
Use `--figures 5` to render only Figure 5, or `--png` to add preview images.

| Figure | Archived inputs under `RESULTS_PAPER/` | Plotted quantity |
|---|---|---|
| 4 | `RQ1_source_confidence_transfer/EXP01_stage3_lodo_main_matrix/Table_3_raw.csv` and its per-seed records | Target group AUROC, averaged over three training seeds |
| 5 | Same RQ1 inputs | Source-internal and target **row-level** AUROC; three-seed method means and equal means across seven methods |
| 6 | Same RQ1 inputs | Target row AUROC versus target group AUROC for the same method-target conditions |
| 7 | `RQ3_target_information_recovery/EXP06_qtransport_paired_source_lr/nine_seed/metrics_long.json` and `EXP07_matched_native_control_attribution/nine_seed/metrics.json` in the same RQ3 directory | Nine paired seed differences per target and control; mean and sample SD |
| 8 | `RQ3_target_information_recovery/EXP08_s0_s3_staircase/staircase_metrics.csv` | S0-S3 target group AUROC, averaged over five support seeds at each target-budget cell |
| 9 | `RQ4_ranking_vs_deployment/EXP09_deployment_operating_audit/coordinate_rows.csv` | Row-level FPR and TPR, averaged over seeds for each target-budget-mode condition |

Figures 1-3 are conceptual diagrams and require no experiment data. Their
artwork is not distributed with this code release.

`tables` writes eight supplementary LaTeX table summaries, two statistical
summaries, and an input/output manifest to `RESULTS/reruns/tables/`.
Both generation commands accept `--output-dir` and keep outputs separate from
`RESULTS_PAPER/`, `DATA/`, `CODE/`, and `paper/`. They do not replace the figures
or text used by the manuscript.
They also accept `--results-root` to render a separately collected experiment
tree. Without this option, they use the supplied `RESULTS_PAPER/` records.

## Obtain and reconstruct data

The four datasets originate from the [UQ NFv2 collection](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets),
which provides NetFlow representations of UNSW-NB15, CSE-CIC-IDS2018, BoT-IoT,
and ToN-IoT. The experiments use **Laurens D'hooge's cleaned Parquet releases
on Kaggle, version 2 (January 2023)**. The NFv2 dataset designation and the
Kaggle release version are separate identifiers; reproduction requires the
specific Kaggle releases listed below.

| Domain | Dataset source on Kaggle | Required Parquet file | Release download |
|---|---|---|---|
| UNSW | [NF-UNSW-NB15-v2](https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2) | `NF-UNSW-NB15-V2.parquet` | [Version 2](https://www.kaggle.com/api/v1/datasets/download/dhoogla/nfunswnb15v2/NF-UNSW-NB15-V2.parquet?datasetVersionNumber=2) |
| CIC | [NF-CSE-CIC-IDS2018-v2](https://www.kaggle.com/datasets/dhoogla/nfcsecicids2018v2) | `NF-CSE-CIC-IDS2018-V2.parquet` | [Version 2](https://www.kaggle.com/api/v1/datasets/download/dhoogla/nfcsecicids2018v2/NF-CSE-CIC-IDS2018-V2.parquet?datasetVersionNumber=2) |
| BoT | [NF-BoT-IoT-v2](https://www.kaggle.com/datasets/dhoogla/nfbotiotv2) | `NF-BoT-IoT-V2.parquet` | [Version 2](https://www.kaggle.com/api/v1/datasets/download/dhoogla/nfbotiotv2/NF-BoT-IoT-V2.parquet?datasetVersionNumber=2) |
| ToN | [NF-ToN-IoT-v2](https://www.kaggle.com/datasets/dhoogla/nftoniotv2) | `NF-ToN-IoT-V2.parquet` | [Version 2](https://www.kaggle.com/api/v1/datasets/download/dhoogla/nftoniotv2/NF-ToN-IoT-V2.parquet?datasetVersionNumber=2) |

To obtain the data:

1. Follow each **Version 2** download link above. If Kaggle requests authentication,
   sign in before downloading. The dataset pages provide the release descriptions.
2. If a download is packaged as a ZIP archive, extract the Parquet file without
   converting or resaving it. Keep the filenames shown in the table.
3. Place all four Parquet files directly in `DATA/nfv2_raw/` under the repository
   root, then run the reconstruction command below.

Raw traffic and processed feature/label arrays are not redistributed. The links
and filenames above match the release records in
[`raw_data_manifest.json`](DATA/nfv2_processed/raw_data_manifest.json), which also
contains the expected SHA-256 fingerprints. The fixed positional indices require
the **original Parquet row order**; do not sort or deduplicate the downloaded
files. An independently converted CSV or newer dataset release is not assumed
equivalent.

```bash
python CODE/run_paper.py prepare-data --output DATA/nfv2_processed
# If arrays already exist, verify without overwriting them:
python CODE/run_paper.py prepare-data --verify
```

The reconstruction checks the raw-file fingerprints and refuses to overwrite
existing arrays. Dataset terms remain those of the providers; the recorded
Parquet releases state CC BY-NC-SA 4.0. Cite the original dataset authors.

## Run experiments

### Q-TRANSPORT and matched controls

Use a separate Python 3.12 or 3.13 environment with
[`requirements-qtransport.txt`](requirements-qtransport.txt). This pins NumPy
2.4.6 and scikit-learn 1.9.1 for the Q/control runners and their CPU dependencies.
From the repository root, with the chosen Python interpreter:

```bash
python -m venv .venv-qtransport
# Linux/macOS: source .venv-qtransport/bin/activate
# Windows PowerShell: .venv-qtransport\Scripts\Activate.ps1
python -m pip install -r requirements-qtransport.txt
python CODE/run_paper.py qtransport
python CODE/run_paper.py matched-control
python CODE/run_paper.py q-summary --q-input RESULTS/reruns/qtransport/metrics_long.json --control-input RESULTS/reruns/matched_control/metrics.json
```

### Source models and RQ2 diagnostics

Use a separate Python 3.10 environment for `requirements-experiments.txt` and
choose the PyTorch 2.1 CPU/CUDA build appropriate for your machine. Original
neural experiments used CUDA 12.1 / RTX 3090; passive-support sensitivity used
RTX 2080 Ti. The requirements record the environment, not a guarantee of
bit-identical neural results on other hardware.

```bash
python -m venv .venv-experiments
# Linux/macOS: source .venv-experiments/bin/activate
# Windows PowerShell: .venv-experiments\Scripts\Activate.ps1
python -m pip install -r requirements-experiments.txt
python CODE/run_paper.py matrix stage3 --dry-run
python CODE/run_paper.py matrix stage3 --device cuda
python CODE/run_paper.py stage3-summary
```

`matrix stage3` runs all 28 method-target conditions: ERM-cw, DANN-source, CORAL,
GroupDRO, IRM, Fishr-lite, and BoDA-style on UNSW, CIC, BoT, and ToN. Each
condition uses the source-validation search (up to eight configurations,
search seeds 42 and 123) followed by final seeds 2026, 2027, and 2028. Outputs,
checkpoints, and prediction caches are placed in `RESULTS/reruns/stage3/`.
Matrices run sequentially and stop on failure; the experiment runners reuse
complete matching records when restarted. `--device cpu` is also supported,
but full neural training can be substantially slower.

Run the model-family controls and diagnostics in the same experiment environment:

```bash
python CODE/run_paper.py tabular
python CODE/run_paper.py matrix anomaly --device cuda
python CODE/run_paper.py summarize families
python CODE/run_paper.py matrix domain-fingerprint --device cuda
python CODE/run_paper.py domain-summary
python CODE/run_paper.py label-conflict
python CODE/run_paper.py label-summary
python CODE/run_paper.py verify-data --output RESULTS/reruns/data_checks
```

The anomaly matrix contains four detectors, four targets, and three seeds.
The fingerprint matrix covers both feature representations, all/benign/attack
domain-prediction tasks, logistic/MLP predictors, and the associated ERM
controls. `label-conflict` consumes those model outputs; `verify-data` consumes
the processed arrays and Stage3 score caches. Neither command retrains Stage3.

### S0–S3, neural adaptation, and operating points

After the complete Stage3 matrix is available, run:

```bash
python CODE/run_paper.py staircase
python CODE/run_paper.py summarize staircase
python CODE/run_paper.py matrix adaptation --device cuda
python CODE/run_paper.py low-budget
python CODE/run_paper.py summarize adaptation
python CODE/run_paper.py deployment-summary
```

The staircase uses seed-2026 scores from all seven source models. The adaptation
matrix crosses four targets, model/checkpoint seeds 2026–2028, and support seeds
11, 23, 37, 53, and 71. Model initialization and optimization use the same model
seed. Each invocation runs Full-FT and Target-only at budgets 50, 100, 500, and
1000, giving 480 adapted coordinates. The original learning-rate grid and
support-validation selection remain in the adaptation runner.

`low-budget` supplies the 240 source-scheduled coordinates at budgets 10 and
20; use `--device cpu` when CUDA is unavailable. `summarize adaptation` also
evaluates the 12 zero-shot source checkpoints on the fixed target test sets.
`deployment-summary` requires the complete 732-coordinate grid and produces
52 target-budget-mode summaries. Ranking and operating-point criteria are
applied after averaging seeds.

### Appendix experiments

Use the experiment environment and retain the Stage3 and adaptation outputs:

```bash
python CODE/run_paper.py support-controls
python CODE/run_paper.py matrix passive-neural --device cuda
python CODE/run_paper.py summarize passive
python CODE/run_paper.py target-calibration
```

`support-controls` uses the supplied fixed acquisition indices to regenerate
balanced, uniform-group, and row-arrival S0–S3 comparisons, including the
single-class fallback. It also evaluates the five alternative ToN partitions
(2029–2033) using fixed source preprocessing and source models. It writes 720
sampling stage records and 100 alternative-partition stage records.

The passive-neural matrix uses 100 uniformly sampled support groups, the same
three model seeds and five support seeds, and both training modes (120
coordinates). Its summary pairs these runs with the balanced N=100 results.

`target-calibration` loads each saved adapted model and selects a threshold
from 100 or 500 additional target training groups. The supplied calibration
indices exclude all adaptation support groups and target test groups. One
representative per group supplies calibration scores; final FPR/TPR are
evaluated per row. No optimization is performed. It produces 960 conditions
and defaults to CPU inference; `--device cuda` can be selected. Optional
`--verify-test-logits` also compares model inference with saved test logits.

### Generate figures and tables from the new runs

After completing all branches above, assemble the new summaries and render them:

```bash
python CODE/run_paper.py collect-results
python CODE/run_paper.py figures --results-root RESULTS/reruns/paper_results --output-dir RESULTS/reruns/new_figures
python CODE/run_paper.py tables --results-root RESULTS/reruns/paper_results --output-dir RESULTS/reruns/new_tables
```

`collect-results` requires every listed input and builds a separate tree at
`RESULTS/reruns/paper_results/`; it does not fill missing experiments with
archived values. Figures 4–9 and the eight supplementary tables then use that
tree. Per-seed records and RQ2/RQ3/RQ4 summaries provide the numerical inputs
for the manuscript's inline tables. `audit` continues to check the supplied
archived evidence. Manuscript comparison additionally requires the local
manuscript files described above.

All matrices support `--dry-run`. `--print-command` previews a `run_paper.py`
dispatch without importing training dependencies. Individual runners expose
their input/output overrides through `--help`; the commands above use the
shared default `RESULTS/reruns/` layout.

Large per-sample prediction caches and checkpoints are excluded from this
compact repository. The sensitivity results include per-condition records,
sampling/calibration manifests, and the arithmetic used to derive the tables.

In `run_qtransport_fair_cpu_v2.py`, label files are opened as read-only memory
maps at cell initialization. For the held-out target, label values are indexed
only during evaluation, after all predictions and their freeze record have been
saved. The legacy `target_labels_loaded: false` field in this runner's
`prediction_freeze.json` means that target label values have not yet been indexed;
it does not mean that the label file has not been opened. Target labels are not
used for quantile fitting, model fitting, or prediction generation.

Two interpretation details matter when reusing the results:

- Balanced support counts selected duplicate groups, not the total labeling cost
  of finding those groups in naturally arriving traffic. It uses candidate-pool labels.
- Legacy deployment fields named `false_alerts_per_million_*` use a denominator
  of one million benign flows. The manuscript's total alert burden uses one million
  all flows. Some legacy group-operating fields contain worst-source row thresholds;
  the paper uses verified row FPR/TPR and group AUROC instead.

## License and citation

Project code and its documentation are released under the [MIT License](LICENSE).
Dataset-derived records are not relicensed by that grant; see
[third-party and non-code notices](THIRD_PARTY_NOTICES.md).

Please cite the manuscript when using this work. [CITATION.cff](CITATION.cff)
provides its authors and the repository citation; no publication DOI is claimed.
