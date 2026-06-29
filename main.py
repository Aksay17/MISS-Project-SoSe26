import pickle
import os
import pandas as pd

from config import SELECTED_DATASETS, MISSING_RATES, RESULTS_DIR
from data.loading import load_data
from preprocessing.cleaning import clean_datasets
from preprocessing.coercion import coerce_types
from missingness.simulation import simulate_all
from imputation.methods import run_all_imputations, impute_complete_case
from analysis.tests import compute_stats, compute_pairwise_stats, split_columns
from evaluation.metrics import (
    compute_all_differences,
    build_summary_df
)
from visualization.plots import (
    plot_heatmap,
    plot_rank,
    plot_degradation_line
)

os.makedirs(RESULTS_DIR, exist_ok=True) ##don't error if it's already there


#Run full simulation + imputation + stats pipeline for one missing rate.
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

    # Load and clean data 
    print("\nStep 1: Loading and cleaning data...")
    dfs         = load_data()
    dfs_cleaned = clean_datasets(dfs)
    dfs_cleaned = coerce_types(dfs_cleaned)

    # Ground truth stats 
    print("\nStep 2: Computing ground truth statistics...")
    ground_truth_stats = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        ground_truth_stats[name] = compute_stats(dfs_cleaned[name])
        print(f"  {name} done")

    # Run pipeline for all missing rates
    print("\nStep 3: Running pipeline for all missing rates...")
    all_rates_results = {}

    for rate in MISSING_RATES:
        mechanisms, imputed_datasets, all_stats = run_pipeline_for_rate(dfs_cleaned, rate)
        all_rates_results[rate] = {
            "mechanisms":       mechanisms,
            "imputed_datasets": imputed_datasets,
            "all_stats":        all_stats
        }

    # save to disk 
    with open(f"{RESULTS_DIR}/all_rates_results.pkl", "wb") as f:
        pickle.dump(all_rates_results, f)
    print(f"\nResults saved to {RESULTS_DIR}/all_rates_results.pkl")

    # Evaluation for primary rate (0.1) 
    print("\nStep 4: Running evaluation for primary rate (0.1)...")
    primary          = all_rates_results[0.1]
    mechanisms       = primary["mechanisms"]
    imputed_datasets = primary["imputed_datasets"]
    all_stats        = primary["all_stats"]

    # differences
    differences = compute_all_differences(all_stats, ground_truth_stats)
    summary_df  = build_summary_df(differences)

    # Plots
    print("\nStep 5: Generating plots...")
    
    for stat_type, value_col in [("pearson", "correlation"), ("anova", "f_stat"), ("chi2", "chi2")]:
        plot_heatmap(stat_type, value_col, summary_df)
        plot_rank(stat_type, differences)

    # degradation line plots (uses all rates)
    all_rates_stats = {
        rate: result["all_stats"]
        for rate, result in all_rates_results.items()
    }
    for stat_type in ["pearson", "anova", "chi2"]:
        plot_degradation_line(stat_type, all_rates_stats, ground_truth_stats)

    print("\nDone.")


if __name__ == "__main__":
    main()