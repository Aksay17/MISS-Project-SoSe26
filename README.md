# Project MISS

Benchmarking missing data imputation strategies on biomedical (TCGA) datasets.

## Structure

```
miss_project/
├── main.py                        # runs everything end to end
├── config.py                      # all settings (missing rates, seeds, etc.)
├── requirements.txt
│
├── data/
│   └── loading.py                 # load TCGA data, split by cancer type
│
├── preprocessing/
│   ├── cleaning.py                # basic_clean, maximize_complete_rows, final_cleanup
│   └── coercion.py                # fix misclassified column types
│
├── missingness/
│   └── simulation.py              # MCAR, MAR, MNAR simulation + restore_dtypes
│
├── imputation/
│   └── methods.py                 # complete_case, mean_mode, knn, mice, missforest
│
├── statistics/
│   └── tests.py                   # split_columns, pearson, anova, chi2 (standard + pairwise)
│
├── evaluation/
│   └── metrics.py                 # abs diff, p-value agreement, NRMSE, F1, Friedman, Nemenyi
│
└── visualization/
    └── plots.py                   # all plots
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

## How to change missingness rates

In `config.py`:
```python
MISSING_RATES = [0.1, 0.2, 0.3]   # add or remove rates here
```

Results are saved to `results/all_rates_results.pkl` so you don't need to rerun
the full pipeline every time. To load saved results:

```python
import pickle
with open("results/all_rates_results.pkl", "rb") as f:
    all_rates_results = pickle.load(f)
```

## Adding a new imputation method

1. Add the function to `imputation/methods.py`
2. Add it to the `imputation_fns` dict in `run_all_imputations()`
3. Rerun `main.py`

## References

- van Buuren & Groothuis-Oudshoorn (2011) — MICE: https://www.jstatsoft.org/article/view/v045i03
- Stekhoven & Bühlmann (2012) — MissForest: https://academic.oup.com/bioinformatics/article/28/1/112/219101
