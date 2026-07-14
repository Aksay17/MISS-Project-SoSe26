# MISS Project:  Missing-Data Imputation Methods on Clinical Cancer Data

A pipeline that evaluates how well different imputation methods cope
with missing values in clinical cancer datasets. For each dataset, missingness is
**simulated** under controlled conditions, filled by several imputation methods,
and the results are compared against the known ground truth at two levels:

1. **Statistic preservation** — how faithfully the imputed data reproduces the
   statistical relationships (correlations, group differences, categorical
   associations) present in the original data.
2. **Value recovery** — how accurately the individual missing values themselves
   are reconstructed.

> Bachelor's/Master's/Project work at the Biomedical Network Science Lab,
> Department Artificial Intelligence in Biomedical Engineering,
> Friedrich-Alexander-Universität Erlangen-Nürnberg.

---

## Overview

The experiment is a full factorial over four axes:

- **4 datasets**: Invasive Breast Carcinoma, Non-Small Cell Lung Cancer,
  Well-Differentiated Thyroid Cancer, Endometrial Carcinoma.
- **3 missingness mechanisms**: MCAR, MAR, MNAR.
- **3 missingness rates**: 10%, 20%, 30%.
- **6 methods**: complete-case deletion, mean/mode, KNN, MICE, MissForest,
  and pairwise (available-case) analysis.


## How it works

The pipeline (`main.py`) runs end to end:

1. **Load** the clinical data and split it by cancer type (`data/loading.py`).
2. **Clean** each dataset; drop empty/constant/ID-like columns and greedily
   reduce columns so that a usable fraction of fully complete rows remains
   (`preprocessing/cleaning.py`).
3. **Coerce** column types (numeric vs. categorical) so the right tests and
   imputers are applied (`preprocessing/coercion.py`).
4. **Simulate** missingness under MCAR/MAR/MNAR at each rate
   (`missingness/simulation.py`).
5. **Impute** with every method (`imputation/methods.py`).
6. **Analyse** compute Pearson correlation, one-way ANOVA, and chi-square
   statistics on each dataset (`analysis/tests.py`).
7. **Evaluate** measure each method's deviation from ground truth, rank the
   methods, and compute cell-level recovery accuracy (`evaluation/metrics.py`).
8. **Plot** the results (`visualization/plots.py`).

## Project structure

```
MISS-Project-SoSe26/
├── config.py                # all settings (paths, rates, seeds, hyperparameters)
├── main.py                  # runs the full pipeline and generates all plots
├── requirements.txt
├── data/
│   ├── loading.py           # loads the TSV and splits it by cancer type
│   └── data_file/           # place the source data here (gitignored)
├── preprocessing/
│   ├── cleaning.py          # column cleaning + complete-row maximisation
│   └── coercion.py          # numeric/categorical type coercion
├── missingness/
│   └── simulation.py        # MCAR/MAR/MNAR simulation (via missmecha)
├── imputation/
│   └── methods.py           # complete_case, mean_mode, knn, mice, missforest
├── analysis/
│   └── tests.py             # Pearson, ANOVA, chi-square, pairwise variants
├── evaluation/
│   └── metrics.py           # statistic differences, rankings, recovery accuracy
├── visualization/
│   └── plots.py             # heatmaps, rank plots, degradation lines, recovery bars
└── plots/                   # generated figures (overwritten on each run)
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data setup

The datasets are derived from publicly available clinical cancer data obtained
from [TCGAclinical](https://github.com/GerkeLab/TCGAclinical). 

1. Obtain the source TSV file. *(cBioportal_data.tsv)*
2. Place it under `data/data_file/`.
3. Point `DATA_PATH` in `config.py` to that file.

## Usage

After installing the dependencies and setting up the data, run:

```bash
python main.py
```

This executes the full pipeline and writes all figures to the `plots/` folder,
overwriting any figures from a previous run.

## Configuration

All settings live in `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `DATA_PATH` | *(local path)* | Location of the source TSV file. |
| `SELECTED_DATASETS` | 4 cancer types | Which cancer types to analyse. |
| `TARGET_ROW_RETENTION` | `0.2` | Minimum fraction of complete rows to keep during cleaning. |
| `MISSING_RATES` | `[0.1, 0.2, 0.3]` | Missingness rates to simulate. |
| `SEED` | `42` | Random seed for reproducibility. |
| `KNN_K` | `5` | Number of neighbours for KNN imputation. |
| `MICE_ITER` | `10` | Iterations for MICE and MissForest. |
| `RF_TREES` | `100` | Trees per learner in MissForest. |
| `MIN_GROUP_SIZE` | `5` | Minimum group size for an ANOVA comparison. |
| `MAX_CARDINALITY` | `10` | Maximum distinct values for a categorical test. |
| `PLOTS_DIR` | `"plots"` | Output folder for figures. |

## Outputs

Running the pipeline produces the following figures in `plots/` (each overwritten
on every run):

- `heatmap_{pearson,anova,chi2}_{MCAR,MAR,MNAR}.png`: absolution mean deviation on each dataset and method, one heatmap per statistic and mechanism (plotted for 10% missingness).
- `rank_{pearson,anova,chi2}.png`: method rankings aggregated across all
  missingness rates, mechanisms, and datasets.
- `degradation_{pearson,anova,chi2}.png`: how deviation grows with the
  missingness rate, one line per method.
- `recovery_bar_nrmse.png`: numeric value-recovery error (NRMSE; lower is
  better), grouped bar by rate, one panel per mechanism.
- `recovery_bar_cat_accuracy.png`: categorical value-recovery accuracy (higher
  is better), same layout.

## Evaluation metrics

- **Statistic difference**: absolute difference between a statistic (Pearson
  correlation, ANOVA F, chi-square) computed on the imputed data and on the
  ground truth.
- **Aggregated ranking**: within each (rate, mechanism, dataset) situation the
  methods are ranked by their statistic difference, and the ranks are then
  averaged across all situations.
- **NRMSE**: root-mean-square error of the imputed numeric values at the masked
  cells, normalised by each column's ground-truth range, then averaged across
  numeric columns. Lower is better.
- **Categorical accuracy**: fraction of masked categorical cells recovered with
  the exact correct label, averaged per column and then across columns. Higher
  is better.

Note: complete-case and pairwise analysis do not fill individual cells, so they
appear in the statistic-based comparisons but not in the value-recovery plots.


