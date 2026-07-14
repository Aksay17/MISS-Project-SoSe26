import os
import glob
import pandas as pd

from config import SELECTED_DATASETS, MISSING_RATES, PLOTS_DIR
from data.loading import load_data
from preprocessing.cleaning import clean_datasets
from preprocessing.coercion import coerce_types
from missingness.simulation import simulate_all
from imputation.methods import run_all_imputations, impute_complete_case
from analysis.tests import compute_stats, compute_pairwise_stats, split_columns
from evaluation.metrics import (
    compute_all_differences,
    build_summary_df,
    compute_recovery_accuracy
)
from visualization.plots import (
    plot_heatmap,
    plot_rank,
    plot_degradation_line,
    plot_recovery_bar
)

#Runs full simulation + imputation + stats pipeline for one missing rate.
def run_pipeline_for_rate(dfs_cleaned, missing_rate):
    print(f"\n{'='*60}")
    print(f"Running pipeline for missing_rate = {missing_rate}")
    print(f"{'='*60}")

    # simulate missingness
    mechanisms = simulate_all(dfs_cleaned, missing_rate)

    # impute
    imputed_datasets = run_all_imputations(mechanisms)

    # compute stats on imputed datasets
    all_stats = {}
    for method_name, mech_datasets in imputed_datasets.items():
        all_stats[method_name] = {}
        for mech, datasets in mech_datasets.items():
            all_stats[method_name][mech] = {}
            for name, df in datasets.items():
                all_stats[method_name][mech][name] = compute_stats(df)
                print(f"  stats: {method_name} | {mech} | {name}")

    # pairwise stats (no imputation — run on missing datasets directly)
    all_stats["pairwise"] = {}
    for mech, datasets in mechanisms.items():
        all_stats["pairwise"][mech] = {}
        for name, df in datasets.items():
            all_stats["pairwise"][mech][name] = compute_pairwise_stats(df)

    return mechanisms, imputed_datasets, all_stats


def main():

    # 1. Load and clean data 
    print("\ni. Loading and cleaning data.")
    dfs         = load_data()
    dfs_cleaned = clean_datasets(dfs)
    dfs_cleaned = coerce_types(dfs_cleaned)

    # 2. Ground truth stats 
    print("\nii. Computing ground truth statistics.")
    ground_truth_stats = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        ground_truth_stats[name] = compute_stats(dfs_cleaned[name])
        print(f"  {name} done")

    # 3. Run pipeline for all missing rates 
    print("\niii. Running pipeline for all missing rates.")
    all_rates_results = {}

    for rate in MISSING_RATES:
        mechanisms, imputed_datasets, all_stats = run_pipeline_for_rate(dfs_cleaned, rate)
        all_rates_results[rate] = {
            "mechanisms":       mechanisms,
            "imputed_datasets": imputed_datasets,
            "all_stats":        all_stats
        }

    # 4. Evaluation for primary rate (0.1)
    print("\niv. Running evaluation for primary rate (0.1).")
    primary          = all_rates_results[0.1]
    mechanisms       = primary["mechanisms"]
    imputed_datasets = primary["imputed_datasets"]
    all_stats        = primary["all_stats"]

    # differences
    differences = compute_all_differences(all_stats, ground_truth_stats)
    summary_df  = build_summary_df(differences)

    # 5. Plots
    print("\nv. Generating plots.")

    # Start from a clean plots folder so each run overwrites the previous
    # figures and stale plots don't accumulate across runs.
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for old_plot in glob.glob(os.path.join(PLOTS_DIR, "*.png")):
        os.remove(old_plot)

    # all_rates_stats spans every missing rate — used by the aggregated rank
    # plot and the degradation line plots.
    all_rates_stats = {
        rate: result["all_stats"]
        for rate, result in all_rates_results.items()
    }

    for stat_type, value_col in [("pearson", "correlation"), ("anova", "f_stat"), ("chi2", "chi2")]:
        plot_heatmap(stat_type, value_col, summary_df)
        plot_rank(stat_type, all_rates_stats, ground_truth_stats)

    # degradation line plots (uses all rates)
    for stat_type in ["pearson", "anova", "chi2"]:
        plot_degradation_line(stat_type, all_rates_stats, ground_truth_stats)

    # cell-level imputation recovery plots
    accuracy_df = compute_recovery_accuracy(all_rates_results, dfs_cleaned)

    # Recovery: grouped bar by rate, one panel per mechanism (bar only)
    plot_recovery_bar(accuracy_df, "nrmse")
    plot_recovery_bar(accuracy_df, "cat_accuracy")

    print("\nDone.")


if __name__ == "__main__":
    main()