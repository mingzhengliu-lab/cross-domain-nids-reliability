# Reliability of Cross-Domain Network Intrusion Detection

Research code and archived evidence for **Reliability of Cross-Domain Network
Intrusion Detection**, by MingZheng Liu, Yiyang Fu, Saru Kumari, and Chien-Ming Chen.

[Manuscript PDF](paper/main_v1.pdf) · [LaTeX source](paper/main_v1.tex) ·
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
python CODE/run_paper.py check-manuscript
```

`audit` verifies the archived coordinate grids, paired seeds, aggregation,
split-isolation records, operating criteria, figure hashes, and references.
`check-manuscript` independently recomputes numerical table entries and headline
statistics against the English manuscript. Reports are written to
`RESULTS/reruns/submission_checks/` and `RESULTS/reruns/manuscript_checks/`.
The initial release passes **51 compact checks and 107 manuscript checks**.
They also pass with Python 3.13, NumPy 2.4.6, Pandas 3.0.3, and scikit-learn 1.9.1.

These commands validate supplied evidence; they do not retrain the models.

## Repository map

| Location | Contents |
|---|---|
| `CODE/` | Experiment runners, preprocessing, aggregation, and verification |
| `RESULTS_PAPER/` | Compact frozen metrics, per-seed records, and provenance |
| `DATA/nfv2_processed/` | Feature definitions, row indices, duplicate-group keys, and fixed splits |
| `paper/` | Current English manuscript, bibliography, figures, and compiled PDF |
| `paper/figs/` | Nine vector PDFs and three editable concept SVGs |

| Research question | Code | Evidence under `RESULTS_PAPER/` |
|---|---|---|
| RQ1: source-to-target transfer | `CODE/experiment_stage3/` | `RQ1_source_confidence_transfer/` |
| RQ2: failure diagnostics | `CODE/experiment_diagnostics/` | `RQ2_failure_heterogeneity/`, `sensitivity/` |
| RQ3: unlabeled Q-TRANSPORT | `CODE/experiment_low_label/` | `RQ3_qtransport_unlabeled_repair/` |
| RQ3: matched-support S0–S3 | `CODE/experiment_staircase/` | `RQ4_low_label_staircase/` |
| RQ4: adaptation and operating points | `CODE/experiment_low_label/` | `RQ5_ranking_vs_deployment/` |

Historical evidence directory names are retained so existing provenance remains
traceable. In particular, the directory numbered RQ4 contains the paper's RQ3
staircase, and the directory numbered RQ5 contains its RQ4 operating-point study.

## Obtain and reconstruct data

The scientific source is the [UQ NFv2 collection](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets).
The experiment uses **Laurens D'hooge's cleaned Parquet releases, version 2
(January 2023)**:

| Domain | Download source |
|---|---|
| UNSW | [NF-UNSW-NB15-v2](https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2) |
| CIC | [NF-CSE-CIC-IDS2018-v2](https://www.kaggle.com/datasets/dhoogla/nfcsecicids2018v2) |
| BoT | [NF-BoT-IoT-v2](https://www.kaggle.com/datasets/dhoogla/nfbotiotv2) |
| ToN | [NF-ToN-IoT-v2](https://www.kaggle.com/datasets/dhoogla/nftoniotv2) |

Raw traffic and processed feature/label arrays are not redistributed. Download
the matching files into `DATA/nfv2_raw/`. Exact filenames, versioned URLs, and
SHA-256 fingerprints are recorded in `DATA/nfv2_processed/raw_data_manifest.json`.
The fixed positional indices require the **original Parquet row order**; an
independently converted CSV or newer dataset release is not assumed equivalent.

```bash
python CODE/run_paper.py prepare-data --output DATA/nfv2_processed
# If arrays already exist, verify without overwriting them:
python CODE/run_paper.py prepare-data --verify
```

The reconstruction checks the raw-file fingerprints and refuses to overwrite
existing arrays. Dataset terms remain those of the providers; the recorded
Parquet releases state CC BY-NC-SA 4.0. Cite the original dataset authors.

## Run experiments

Use a separate Python 3.10 environment for `requirements-experiments.txt` and
choose the PyTorch 2.1 CPU/CUDA build appropriate for your machine. Original
neural experiments used CUDA 12.1 / RTX 3090; passive-support sensitivity used
RTX 2080 Ti. The requirements record the environment, not a guarantee of
bit-identical neural results on other hardware.

```bash
python CODE/run_paper.py stage3 --method erm --target unsw --mode all
python CODE/run_paper.py qtransport
python CODE/run_paper.py matched-control
python CODE/run_paper.py staircase --score-root RESULTS/reruns/stage3
python CODE/run_paper.py low-budget --stage3-root RESULTS/reruns/stage3
```

Stage3 must be run for all seven methods and four targets before the complete
staircase can use the seed-2026 score caches. Full adaptation additionally uses
the corresponding trained checkpoints; inspect its options with
`python CODE/run_paper.py adaptation --help`. Model seeds are 2026–2028 and
support seeds are 11, 23, 37, 53, and 71. `--print-command` previews any runner
without importing training dependencies. Fresh outputs go to `RESULTS/reruns/`.

The nine-seed Q/control matrix was reproduced across all four targets: 684 method
records and 3,312 metric comparisons matched the archive, as did the 72 prediction
and 72 source-index archive hashes. The report is
`RESULTS_PAPER/sensitivity/q_full_matrix_reproduction.json`; the exact-match rerun
used Python 3.13.13, NumPy 2.4.6, and scikit-learn 1.9.1. Large per-sample caches
and checkpoints are excluded from this compact repository. Sampling/calibration
sensitivity is distributed as archived coordinates and checks, rather than a
promise that every historical scheduling script is present.

Two interpretation details matter when reusing the results:

- Balanced support counts selected duplicate groups, not the total labeling cost
  of finding those groups in naturally arriving traffic. It uses candidate-pool labels.
- Legacy deployment fields named `false_alerts_per_million_*` use a denominator
  of one million benign flows. The manuscript's total alert burden uses one million
  all flows. Some legacy group-operating fields contain worst-source row thresholds;
  the paper uses verified row FPR/TPR and group AUROC instead.

## License and citation

Project code and its documentation are released under the [MIT License](LICENSE).
The manuscript, figure artwork, dataset-derived records, and Elsevier CAS files
are not relicensed by that grant; see [third-party and non-code notices](THIRD_PARTY_NOTICES.md).

Please cite the manuscript when using this work. [CITATION.cff](CITATION.cff)
provides its authors and the repository citation; no publication DOI is claimed.
