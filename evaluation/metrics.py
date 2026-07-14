import numpy as np
import pandas as pd

#  Compute absolute difference between imputed and ground truth statistics
#  for every variable pair, mechanism and dataset.
def compute_stat_differences(imputed_stats, ground_truth_stats, stat_type, value_col):
    
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
            #difference between imputed and ground truth values, absolute value
            merged["abs_diff"]  = (merged[f"{value_col}_imputed"] - merged[f"{value_col}_gt"]).abs()
            merged["mechanism"] = mech
            merged["dataset"]   = name

            results.append(merged[["abs_diff", "mechanism", "dataset"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

#Compute differences for all methods and stat types.
def compute_all_differences(all_stats, ground_truth_stats):
    differences = {}
    for method_name, stats_dict in all_stats.items():
        differences[method_name] = {
            "pearson": compute_stat_differences(stats_dict, ground_truth_stats, "pearson", "correlation"),
            "anova":   compute_stat_differences(stats_dict, ground_truth_stats, "anova",   "f_stat"),
            "chi2":    compute_stat_differences(stats_dict, ground_truth_stats, "chi2",    "chi2")
        }
        print(f"  {method_name} differences computed")
    return differences

#Aggregate pair-level differences into mean per method/mechanism/dataset.
def build_summary_df(differences):
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


# Cell-level imputation accuracy at the masked (originally-missing) positions.
# Numeric  -> NRMSE (RMSE normalised by ground-truth range), lower = better.
# Categorical -> proportion of masked cells recovered exactly, higher = better.
# Only value-filling methods are scored (mean_mode, knn, mice, missforest);
# complete_case deletes rows and pairwise never imputes, so both are skipped.
def compute_recovery_accuracy(all_rates_results, dfs_cleaned):
    from analysis.tests import split_columns

    SKIP = {"complete_case"}   # does not impute cell values, pairwise is not in the imputed ds 
    rows = []

    #Loop over each missingness rate (0.1, 0.2, 0.3). For that rate, 
    #pull out the missingness datasets (mechanisms) and the imputed datasets 
    for rate, result in all_rates_results.items():
        mechanisms       = result["mechanisms"]
        imputed_datasets = result["imputed_datasets"]

        #Loop over each imputation method; skip complete_case entirely.
        for method, mech_data in imputed_datasets.items():
            if method in SKIP:
                continue
            for mech, datasets in mech_data.items():
                for name, imp in datasets.items():
                    if name not in dfs_cleaned:
                        continue

                    # Imputers rebuild the frame with a fresh index, 
                    # aligning by row position (not by the old index labels) 
                    # is what keeps the same physical row lined up across all three.
                    truth   = dfs_cleaned[name].reset_index(drop=True)
                    missing = mechanisms[mech][name].reset_index(drop=True)
                    imputed = imp.reset_index(drop=True)
                    #Determine which columns are numeric and which are categorical, based on the ground truth.
                    num_cols, cat_cols = split_columns(truth)

                    # Prepare a list of per-column NRMSE values, 
                    # and loop over numeric columns; skip any that aren't in the imputed frame.
                    nrmse_vals = []
                    for col in num_cols:
                        if col not in imputed.columns:
                            continue
                        #cells that were deleted in this column. 
                        #If nothing was masked here, there's nothing to score; skip.
                        mask = missing[col].isna()
                        if mask.sum() == 0:
                            continue
                        true_v = pd.to_numeric(truth.loc[mask, col],   errors="coerce").to_numpy(float)
                        pred_v = pd.to_numeric(imputed.loc[mask, col], errors="coerce").to_numpy(float)
                        col_range = truth[col].max() - truth[col].min()

                        #Compute the column's range (max − min) to use as the normalizer. 
                        #If the range is valid, use it; if it's zero or NaN, 
                        #fall back to the standard deviation. 
                        #If even that is zero/NaN, skip the column (can't normalize).
                        denom = col_range if (col_range and not np.isnan(col_range)) else truth[col].std()
                        if not denom or np.isnan(denom):
                            continue
                        #square the errors, average them, square-root
                        rmse = np.sqrt(np.nanmean((pred_v - true_v) ** 2))
                        nrmse_vals.append(rmse / denom)

                    # categorical: accuracy per column, then average 
                    acc_vals = []
                    for col in cat_cols:
                        if col not in imputed.columns:
                            continue
                        mask = missing[col].isna()
                        if mask.sum() == 0:
                            continue
                        t = truth.loc[mask, col].astype(str).to_numpy()
                        p = imputed.loc[mask, col].astype(str).to_numpy()
                        acc_vals.append((t == p).mean())

                    rows.append({
                        "rate": rate, "mechanism": mech, "dataset": name, "method": method,
                        "nrmse":        np.mean(nrmse_vals) if nrmse_vals else np.nan,
                        "cat_accuracy": np.mean(acc_vals)   if acc_vals   else np.nan,
                    })

    return pd.DataFrame(rows)