# ============================================================
# evaluation/metrics.py
# All evaluation metrics: abs diff, NRMSE, F1
# ============================================================

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import f1_score, mean_squared_error
from analysis.tests import split_columns, compute_stats, compute_pairwise_stats
#from config import ALPHA, SELECTED_DATASETS


# ── Stat differences ─────────────────────────────────────────

def compute_stat_differences(imputed_stats, ground_truth_stats, stat_type, value_col):
    """
    Compute absolute difference between imputed and ground truth statistics
    for every variable pair, mechanism and dataset.
    """
    results = []

    merge_cols = (
        ["var1", "var2"]              if stat_type in ["pearson", "chi2"]
        else ["categorical", "numerical"]
    )

    for mech, datasets in imputed_stats.items():
        for name, stats in datasets.items():

            imputed_df = stats[stat_type].copy()
            gt_df      = ground_truth_stats[name][stat_type].copy()

            # Merge imputed vs ground-truth stats. If the imputed side produced
            # nothing (e.g. complete-case / listwise deletion wiped out almost
            # all rows under MCAR/MAR), the merge is empty.
            if imputed_df.empty or gt_df.empty:
                merged = pd.DataFrame()
            else:
                merged = imputed_df.merge(gt_df, on=merge_cols, suffixes=("_imputed", "_gt"))

            # Record the cell as a failure (abs_diff = NaN) instead of silently
            # skipping it, so the method stays visible in every plot rather than
            # disappearing entirely.
            if merged.empty:
                results.append(pd.DataFrame(
                    {"abs_diff": [np.nan], "mechanism": [mech], "dataset": [name]}
                ))
                continue

            merged["abs_diff"]  = (merged[f"{value_col}_imputed"] - merged[f"{value_col}_gt"]).abs()
            merged["mechanism"] = mech
            merged["dataset"]   = name

            results.append(merged[["abs_diff", "mechanism", "dataset"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_all_differences(all_stats, ground_truth_stats):
    """Compute differences for all methods and stat types."""
    differences = {}
    for method_name, stats_dict in all_stats.items():
        differences[method_name] = {
            "pearson": compute_stat_differences(stats_dict, ground_truth_stats, "pearson", "correlation"),
            "anova":   compute_stat_differences(stats_dict, ground_truth_stats, "anova",   "f_stat"),
            "chi2":    compute_stat_differences(stats_dict, ground_truth_stats, "chi2",    "chi2")
        }
        print(f"  {method_name} differences computed")
    return differences


def build_summary_df(differences):
    """Aggregate pair-level differences into mean per method/mechanism/dataset."""
    summary = []
    for method_name, stat_diffs in differences.items():
        for stat_type, df in stat_diffs.items():
            if df.empty:
                continue
            grouped = df.groupby(["mechanism", "dataset"])["abs_diff"].mean().reset_index()
            grouped["method"]    = method_name
            grouped["stat_type"] = stat_type
            summary.append(grouped)
    return pd.concat(summary, ignore_index=True) if summary else pd.DataFrame()