# ============================================================
# evaluation/metrics.py
# All evaluation metrics: abs diff, NRMSE, F1
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from analysis.tests import split_columns, compute_stats, compute_pairwise_stats
from config import ALPHA, SELECTED_DATASETS


# ── Stat differences ─────────────────────────────────────────

def compute_stat_differences(imputed_stats, ground_truth_stats, stat_type, value_col):
    """
    Compute absolute difference between imputed and ground truth statistics
    for every variable pair, mechanism and dataset.
    """
    results = []

    for mech, datasets in imputed_stats.items():
        for name, stats in datasets.items():

            imputed_df = stats[stat_type].copy()
            gt_df      = ground_truth_stats[name][stat_type].copy()

            if imputed_df.empty or gt_df.empty:
                continue

            merge_cols = (
                ["var1", "var2"]
                if stat_type in ["pearson", "chi2"]
                else ["categorical", "numerical"]
            )

            merged = imputed_df.merge(gt_df, on=merge_cols, suffixes=("_imputed", "_gt"))

            if merged.empty:
                continue

            merged["abs_diff"]  = (merged[f"{value_col}_imputed"] - merged[f"{value_col}_gt"]).abs()
            merged["mechanism"] = mech
            merged["dataset"]   = name

            results.append(merged[merge_cols + ["abs_diff", "mechanism", "dataset"]])

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


# ── NRMSE and F1 ─────────────────────────────────────────────

def compute_imputation_metrics(imputed_datasets, mechanisms, dfs_cleaned):
    """
    Compute NRMSE (numeric columns) and F1 (categorical columns)
    by comparing imputed values to original values at missing positions.

    Complete case drops rows — we preserve the original index from dropna()
    and use it to align before resetting, so we only evaluate on surviving rows.
    All indexing uses numpy arrays to avoid pandas index misalignment issues.
    """
    results = []

    for method_name, mech_datasets in imputed_datasets.items():
        for mech, datasets in mech_datasets.items():
            for name, df_imputed in datasets.items():

                df_original = dfs_cleaned[name].reset_index(drop=True)
                df_missing  = mechanisms[mech][name].reset_index(drop=True)
                # do NOT reset df_imputed yet — preserve its index for alignment

                if df_original.shape[0] != df_missing.shape[0]:
                    print(f"  WARNING: shape mismatch — {method_name} | {mech} | {name}")
                    continue

                # complete case drops rows — align using preserved index, then reset
                if df_imputed.shape[0] != df_original.shape[0]:
                    shared_idx  = df_imputed.index.intersection(df_original.index)
                    df_original = df_original.loc[shared_idx].reset_index(drop=True)
                    df_missing  = df_missing.loc[shared_idx].reset_index(drop=True)
                    df_imputed  = df_imputed.reset_index(drop=True)
                else:
                    df_imputed  = df_imputed.reset_index(drop=True)

                num_cols, cat_cols = split_columns(df_original)

                for col in df_original.columns:
                    if col not in df_imputed.columns or col not in df_missing.columns:
                        continue

                    # use numpy arrays throughout to avoid all index alignment issues
                    missing_mask = df_missing[col].isna().to_numpy()
                    if missing_mask.sum() == 0:
                        continue

                    orig    = df_original[col].to_numpy()
                    imputed = df_imputed[col].to_numpy()

                    orig_at_missing    = orig[missing_mask]
                    imputed_at_missing = imputed[missing_mask]

                    if col in num_cols:
                        orig_vals    = pd.to_numeric(pd.Series(orig_at_missing),    errors="coerce")
                        imputed_vals = pd.to_numeric(pd.Series(imputed_at_missing), errors="coerce")
                        valid        = orig_vals.notna()
                        orig_vals    = orig_vals[valid]
                        imputed_vals = imputed_vals[valid]

                        if len(orig_vals) < 2:
                            continue

                        rmse    = np.sqrt(mean_squared_error(orig_vals, imputed_vals))
                        col_std = pd.Series(orig).std()
                        nrmse   = rmse / col_std if col_std > 0 else np.nan

                        if pd.isna(nrmse):
                            continue

                        results.append({"method": method_name, "mechanism": mech, "dataset": name,
                                        "column": col, "type": "numeric", "metric": "NRMSE", "value": nrmse})

                    elif col in cat_cols:
                        orig_vals    = pd.Series(orig_at_missing).astype(str)
                        imputed_vals = pd.Series(imputed_at_missing).astype(str)

                        valid_mask   = orig_vals != "nan"
                        orig_vals    = orig_vals[valid_mask]
                        imputed_vals = imputed_vals[valid_mask]

                        if len(orig_vals) < 2 or orig_vals.nunique() < 2:
                            continue

                        known_classes = set(orig_vals.unique())
                        imputed_vals  = imputed_vals.apply(
                            lambda x: x if x in known_classes else orig_vals.mode()[0]
                        )

                        f1 = f1_score(orig_vals, imputed_vals, average="weighted", zero_division=0)

                        if not (0 <= f1 <= 1):
                            continue

                        results.append({"method": method_name, "mechanism": mech, "dataset": name,
                                        "column": col, "type": "categorical", "metric": "F1", "value": f1})

    return pd.DataFrame(results)