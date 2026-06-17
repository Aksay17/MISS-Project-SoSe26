import os
from config import SELECTED_DATASETS, MISSING_RATES, PLOTS_DIR
from data.loading import load_data
from preprocessing.cleaning import clean_datasets
from preprocessing.coercion import coerce_types
from missingness.simulation import simulate_all
from imputation.methods import run_all_imputations
from analysis.tests import compute_stats, compute_pairwise_stats
from evaluation.metrics import (
    compute_all_differences,
    build_summary_df,
    compute_imputation_metrics
)
from visualization.plots import (
    plot_heatmap,
    plot_rank,
    plot_metrics,
    plot_degradation_line,
    plot_degradation_aggregated,
)

os.makedirs(PLOTS_DIR, exist_ok=True)


def run_pipeline_for_rate(dfs_cleaned, missing_rate):
    """Run full simulation + imputation + stats pipeline for one missing rate."""
    print(f"\n{'='*60}")
    print(f"Running pipeline for missing_rate = {missing_rate}")
    print(f"{'='*60}")

    mechanisms       = simulate_all(dfs_cleaned, missing_rate)
    imputed_datasets = run_all_imputations(mechanisms)

    all_stats = {}
    for method_name, mech_datasets in imputed_datasets.items():
        all_stats[method_name] = {}
        for mech, datasets in mech_datasets.items():
            all_stats[method_name][mech] = {}
            for name, df in datasets.items():
                all_stats[method_name][mech][name] = compute_stats(df)
                print(f"  stats: {method_name} | {mech} | {name}")

    all_stats["pairwise"] = {}
    for mech, datasets in mechanisms.items():
        all_stats["pairwise"][mech] = {}
        for name, df in datasets.items():
            all_stats["pairwise"][mech][name] = compute_pairwise_stats(df)

    return mechanisms, imputed_datasets, all_stats


def main():

    # ── 1. Load and clean ────────────────────────────────────
    print("\n[1] Loading and cleaning data...")
    dfs         = load_data()
    dfs_cleaned = clean_datasets(dfs)
    dfs_cleaned = coerce_types(dfs_cleaned)

    # ── 2. Ground truth stats ────────────────────────────────
    print("\n[2] Computing ground truth statistics...")
    ground_truth_stats = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        ground_truth_stats[name] = compute_stats(dfs_cleaned[name])
        print(f"  {name} done")

    # ── 3. Run pipeline for all missing rates ────────────────
    print("\n[3] Running pipeline for all missing rates...")
    all_rates_results = {}

    for rate in MISSING_RATES:
        mechanisms, imputed_datasets, all_stats = run_pipeline_for_rate(dfs_cleaned, rate)
        all_rates_results[rate] = {
            "mechanisms":       mechanisms,
            "imputed_datasets": imputed_datasets,
            "all_stats":        all_stats
        }

    # ── 4. Evaluation ────────────────────────────────────────
    print("\n[4] Running evaluation for primary rate (0.1)...")
    primary          = all_rates_results[0.1]
    mechanisms       = primary["mechanisms"]
    imputed_datasets = primary["imputed_datasets"]
    all_stats        = primary["all_stats"]

    differences = compute_all_differences(all_stats, ground_truth_stats)
    summary_df  = build_summary_df(differences)
    metrics_df  = compute_imputation_metrics(imputed_datasets, mechanisms, dfs_cleaned)

    all_rates_stats = {
        rate: result["all_stats"]
        for rate, result in all_rates_results.items()
    }

    # ── 5. Save plots ─────────────────────────────────────────
    print(f"\n[5] Saving plots to {PLOTS_DIR}/...")

    for stat_type, value_col in [("pearson", "correlation"), ("anova", "f_stat"), ("chi2", "chi2")]:
        plot_heatmap(stat_type, value_col, summary_df,            save_dir=PLOTS_DIR)
        plot_rank(stat_type, differences,                         save_dir=PLOTS_DIR)
        plot_degradation_line(stat_type, all_rates_stats,
                              ground_truth_stats,                 save_dir=PLOTS_DIR)
        plot_degradation_aggregated(stat_type, all_rates_stats,
                                    ground_truth_stats,           save_dir=PLOTS_DIR)

    plot_metrics("NRMSE", metrics_df, save_dir=PLOTS_DIR)
    plot_metrics("F1",    metrics_df, save_dir=PLOTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()