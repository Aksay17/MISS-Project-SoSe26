# ============================================================
# visualization/plots.py
# All plotting functions.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ALPHA, PLOTS_DIR


# ── Heatmaps ─────────────────────────────────────────────────

def plot_heatmap(stat_type, value_col, summary_df):
    """Mean absolute difference heatmap — rows=datasets, cols=methods."""
    df = summary_df[summary_df["stat_type"] == stat_type].copy()

    if df.empty:
        print(f"No data for {stat_type}")
        return

    for mech in ["MCAR", "MAR", "MNAR"]:
        mech_df = df[df["mechanism"] == mech].pivot(
            index="dataset", columns="method", values="abs_diff"
        )
        if mech_df.empty:
            continue

        plt.figure(figsize=(14, 6))
        ax = sns.heatmap(mech_df, annot=True, fmt=".4f", cmap="YlOrRd", linewidths=0.5,
                         cbar_kws={"label": f"Mean Absolute Difference ({value_col})"})

        # Cells with no data (e.g. complete-case under MCAR/MAR) are NaN, which
        # seaborn leaves blank. Shade them and overlay an explicit "N/A" so the
        # method stays visible and the failure is unambiguous.
        ax.set_facecolor("#eeeeee")
        for yi, dataset in enumerate(mech_df.index):
            for xi, method in enumerate(mech_df.columns):
                if pd.isna(mech_df.iloc[yi, xi]):
                    ax.text(xi + 0.5, yi + 0.5, "N/A", ha="center", va="center",
                            fontsize=11, color="grey", fontweight="bold")
        plt.title(f"{stat_type.upper()} - {mech}: Mean Absolute Difference from Ground Truth",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Imputation Method", fontsize=12)
        plt.ylabel("Cancer Dataset", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, f"heatmap_{stat_type}_{mech}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()

# ── Rank plot ────────────────────────────────────────────────

def plot_rank(stat_type, differences):
    """Average rank plot — lower rank = better method."""
    all_data = []
    for method_name, stat_diffs in differences.items():
        df = stat_diffs[stat_type]
        if df.empty:
            continue
        df = df.copy()
        df["method"] = method_name
        all_data.append(df)

    if not all_data:
        print(f"No data for {stat_type}")
        return

    combined = pd.concat(all_data, ignore_index=True)
    grouped  = combined.groupby(["mechanism", "dataset", "method"])["abs_diff"].mean().reset_index()
    grouped["rank"] = grouped.groupby(["mechanism", "dataset"])["abs_diff"].rank(
        method="min", ascending=True
    )

    avg_rank  = grouped.groupby("method")["rank"].mean().reset_index().sort_values("rank")
    mech_rank = grouped.groupby(["mechanism", "method"])["rank"].mean().reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(28, 8))
    colors    = {"MCAR": "#55A868", "MAR": "#DD8452", "MNAR": "#C44E52"}

    # overall
    ax   = axes[0]
    bars = ax.barh(avg_rank["method"], avg_rank["rank"], color="#4C72B0", alpha=0.8)
    ax.set_title("Overall Average Rank", fontsize=16, fontweight="bold")
    ax.set_xlabel("Average Rank (lower = better)", fontsize=13)
    ax.set_xlim(0, 5.5)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, avg_rank["rank"]):
        if pd.isna(val):
            continue
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=12)

    # per mechanism
    for i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
        ax      = axes[i+1]
        mech_df = mech_rank[mech_rank["mechanism"] == mech].sort_values("rank")
        bars    = ax.barh(mech_df["method"], mech_df["rank"], color=colors[mech], alpha=0.8)
        ax.set_title(f"{mech} Average Rank", fontsize=16, fontweight="bold")
        ax.set_xlabel("Average Rank (lower = better)", fontsize=13)
        ax.set_xlim(0, 5.5)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, mech_df["rank"]):
            if pd.isna(val):
                continue
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=12)

    fig.suptitle(f"{stat_type.upper()}: Method Rankings by Average Performance",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"rank_{stat_type}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ── Line plot for missingness rate degradation ───────────────

def plot_degradation_line(stat_type, all_rates_stats, ground_truth_stats,
                          min_coverage=30):
    """
    Line plot showing performance degradation across missingness rates.
    x = missing rate, y = mean absolute difference, one line per method.

    Coverage guard
    --------------
    Each plotted point is the mean of N per-pair differences. Under heavy
    missingness, complete-case deletion can leave so few rows that only a
    handful of tests survive (and on different datasets at each rate), making
    the point unstable and not comparable across x. Any point backed by fewer
    than `min_coverage` valid tests is set to NaN so it drops out of the line
    instead of being plotted as a misleading value. Set min_coverage=0 to
    disable the guard.
    """
    from evaluation.metrics import compute_stat_differences

    value_col = (
        "correlation" if stat_type == "pearson" else
        "f_stat"      if stat_type == "anova"   else
        "chi2"
    )

    rates   = sorted(all_rates_stats.keys())
    methods = list(next(iter(all_rates_stats.values())).keys())
    colors  = {"complete_case": "#937860", "mean_mode": "#4C72B0", "knn": "#DD8452",
               "mice": "#55A868", "missforest": "#C44E52", "pairwise": "#8172B2"}

    fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharey=False)

    for ax, mech in zip(axes, ["MCAR", "MAR", "MNAR"]):
        for method in methods:
            y_vals = []
            for rate in rates:
                diffs = compute_stat_differences(
                    {mech: all_rates_stats[rate][method][mech]},
                    ground_truth_stats, stat_type, value_col
                )
                vals = (diffs["abs_diff"].dropna()
                        if not diffs.empty else pd.Series(dtype=float))
                # Drop points backed by too few tests to be trustworthy.
                if len(vals) < min_coverage:
                    y_vals.append(np.nan)
                else:
                    y_vals.append(vals.mean())

            ax.plot([r * 100 for r in rates], y_vals,
                    marker="o", label=method, color=colors.get(method, "gray"),
                    linewidth=2, markersize=8)

        ax.set_title(f"{mech}", fontsize=18, fontweight="bold")
        ax.set_xlabel("Missing Rate (%)", fontsize=14)
        ax.set_ylabel("Mean Absolute Difference", fontsize=14)
        ax.set_xticks([r * 100 for r in rates])
        ax.legend(title="Method", fontsize=12)
        ax.grid(linestyle="--", alpha=0.5)

    fig.suptitle(f"{stat_type.upper()}: Performance Degradation by Missingness Rate",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"degradation_{stat_type}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()