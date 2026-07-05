# ============================================================
# regenerate_plots.py
# Regenerate all plots from a previously saved results pickle,
# WITHOUT rerunning the simulation / imputation pipeline.
#
# Place this file in the project root (next to main.py) and run:
#     python regenerate_plots.py
#
# Ground truth is not stored in the pickle, so it is recomputed
# from the source data (this is deterministic — same data + same
# config produces the same cleaned datasets every time).
# ============================================================

import os
import sys
import pickle

# Save figures to disk instead of opening windows. Set to False if you
# are running interactively (e.g. in Jupyter) and want plt.show() instead.
SAVE_PLOTS = True
PLOTS_OUTPUT_DIR = "plots"

# Resolve paths relative to this script so it works from any working dir.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if SAVE_PLOTS:
    import matplotlib
    matplotlib.use("Agg")          # headless backend — must be set before pyplot
import matplotlib.pyplot as plt

import config
from config import SELECTED_DATASETS, RESULTS_DIR

# ── Make the data path portable ─────────────────────────────
# config.DATA_PATH may be an absolute path from another machine. Fall back to
# the TSV bundled in the project if that path doesn't exist here.
_local_tsv = os.path.join(PROJECT_ROOT, "data", "data_file", "cBioportal_data.tsv")
if not os.path.exists(config.DATA_PATH) and os.path.exists(_local_tsv):
    config.DATA_PATH = _local_tsv
    import data.loading as _loading
    _loading.DATA_PATH = _local_tsv     # the module imported it by value

from data.loading import load_data
from preprocessing.cleaning import clean_datasets
from preprocessing.coercion import coerce_types
from analysis.tests import compute_stats
from evaluation.metrics import (
    compute_all_differences,
    build_summary_df,
    compute_recovery_accuracy
)
from visualization import plots


def recompute_ground_truth():
    """
    Rebuild ground-truth stats AND the cleaned datasets from source data
    (deterministic). dfs_cleaned is needed for the cell-level recovery metric.
    """
    print("[1] Recomputing ground truth from source data...")
    dfs         = load_data()
    dfs_cleaned = clean_datasets(dfs)
    dfs_cleaned = coerce_types(dfs_cleaned)
    ground_truth_stats = {
        name: compute_stats(dfs_cleaned[name])
        for name in SELECTED_DATASETS if name in dfs_cleaned
    }
    return ground_truth_stats, dfs_cleaned


def load_results():
    """Load the saved pipeline results pickle."""
    path = os.path.join(RESULTS_DIR, "all_rates_results.pkl")
    print(f"[2] Loading saved results from {path} ...")
    with open(path, "rb") as f:
        return pickle.load(f)


def _make_show_saver():
    """Replace plt.show() with a saver that writes each figure to PLOTS_OUTPUT_DIR."""
    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    state = {"i": 0, "name": "plot"}

    def _save(*args, **kwargs):
        state["i"] += 1
        fig = plt.gcf()
        out = os.path.join(PLOTS_OUTPUT_DIR, f"{state['i']:02d}_{state['name']}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"      saved {out}")

    plt.show = _save
    return state


def main():
    ground_truth_stats, dfs_cleaned = recompute_ground_truth()
    all_rates_results = load_results()

    primary   = all_rates_results[0.1]      # primary missing rate
    all_stats = primary["all_stats"]

    print("[3] Computing differences...")
    differences = compute_all_differences(all_stats, ground_truth_stats)
    summary_df  = build_summary_df(differences)

    # If saving, label each figure by hijacking plt.show().
    label = _make_show_saver() if SAVE_PLOTS else None

    print("[4] Generating plots...")
    all_rates_stats = {rate: r["all_stats"] for rate, r in all_rates_results.items()}
    for stat_type, value_col in [("pearson", "correlation"),
                                 ("anova", "f_stat"),
                                 ("chi2", "chi2")]:
        if label: label["name"] = f"heatmap_{stat_type}"
        plots.plot_heatmap(stat_type, value_col, summary_df)
        if label: label["name"] = f"rank_{stat_type}"
        plots.plot_rank(stat_type, all_rates_stats, ground_truth_stats)

    for stat_type in ["pearson", "anova", "chi2"]:
        if label: label["name"] = f"degradation_{stat_type}"
        plots.plot_degradation_line(stat_type, all_rates_stats, ground_truth_stats)

    # cell-level imputation recovery plots
    print("[5] Generating recovery plots...")
    accuracy_df = compute_recovery_accuracy(all_rates_results, dfs_cleaned)

    # Recovery: grouped bar only (by rate, one panel per mechanism)
    if label: label["name"] = "recovery_bar_nrmse"
    plots.plot_recovery_bar(accuracy_df, "nrmse")
    if label: label["name"] = "recovery_bar_cat_accuracy"
    plots.plot_recovery_bar(accuracy_df, "cat_accuracy")

    if SAVE_PLOTS:
        print(f"\nDone. Plots written to ./{PLOTS_OUTPUT_DIR}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()