# ============================================================
# visualization/plots.py
# All plotting functions.
# Each function accepts a save_path argument; pass a filepath
# to save to disk instead of displaying interactively.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ALPHA


def _save_or_show(save_path):
    """Save figure to disk if save_path given, otherwise show interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path}")
        plt.close()
    else:
        plt.show()


# ── Heatmaps ─────────────────────────────────────────────────

def plot_heatmap(stat_type, value_col, summary_df, save_dir=None):
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
        sns.heatmap(mech_df, annot=True, fmt=".4f", cmap="YlOrRd", linewidths=0.5,
                    cbar_kws={"label": f"Mean Absolute Difference ({value_col})"})
        plt.title(f"{stat_type.upper()} - {mech}: Mean Absolute Difference from Ground Truth",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Imputation Method", fontsize=12)
        plt.ylabel("Cancer Dataset", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"heatmap_{stat_type}_{mech}.png") if save_dir else None
        _save_or_show(save_path)


# ── Rank plot ────────────────────────────────────────────────

def plot_rank(stat_type, differences, save_dir=None):
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

    ax   = axes[0]
    bars = ax.barh(avg_rank["method"], avg_rank["rank"], color="#4C72B0", alpha=0.8)
    ax.set_title("Overall Average Rank", fontsize=16, fontweight="bold")
    ax.set_xlabel("Average Rank (lower = better)", fontsize=13)
    ax.set_xlim(0, 6.5)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, avg_rank["rank"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=12)

    for i, mech in enumerate(["MCAR", "MAR", "MNAR"]):
        ax      = axes[i+1]
        mech_df = mech_rank[mech_rank["mechanism"] == mech].sort_values("rank")
        bars    = ax.barh(mech_df["method"], mech_df["rank"], color=colors[mech], alpha=0.8)
        ax.set_title(f"{mech} Average Rank", fontsize=16, fontweight="bold")
        ax.set_xlabel("Average Rank (lower = better)", fontsize=13)
        ax.set_xlim(0, 6.5)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, mech_df["rank"]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=12)

    fig.suptitle(f"{stat_type.upper()}: Method Rankings by Average Performance",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"rank_{stat_type}.png") if save_dir else None
    _save_or_show(save_path)


# ── NRMSE / F1 bar charts ────────────────────────────────────

def plot_metrics(metric_name, metrics_df, save_dir=None):
    """Bar chart of NRMSE or F1 per method per mechanism per dataset."""
    df = metrics_df[metrics_df["metric"] == metric_name].copy()

    if df.empty:
        print(f"No data for {metric_name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharey=False)

    for ax, mech in zip(axes, ["MCAR", "MAR", "MNAR"]):
        mech_df  = df[df["mechanism"] == mech]
        if mech_df.empty:
            ax.set_title(f"{mech} - No data", fontsize=16)
            continue

        methods  = mech_df["method"].unique()
        datasets = mech_df["dataset"].unique()
        x        = np.arange(len(methods))
        width    = 0.8 / len(datasets)
        colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for i, (dataset, color) in enumerate(zip(datasets, colors)):
            ds_df  = mech_df[mech_df["dataset"] == dataset]
            values = [ds_df[ds_df["method"] == m]["value"].mean() for m in methods]
            offset = (i - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=dataset, color=color, alpha=0.8)

        ax.set_title(f"{mech}", fontsize=18, fontweight="bold")
        ax.set_xlabel("Imputation Method", fontsize=14)
        ax.set_ylabel(metric_name, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=13)
        ax.legend(title="Dataset", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"{metric_name}: Imputation Accuracy by Method and Mechanism",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"metrics_{metric_name}.png") if save_dir else None
    _save_or_show(save_path)


# ── Degradation — split by mechanism ─────────────────────────

def plot_degradation_line(stat_type, all_rates_stats, ground_truth_stats, save_dir=None):
    """
    Line plot: degradation across missing rates.
    One subplot per mechanism, one line per method.
    """
    from evaluation.metrics import compute_stat_differences

    value_col = (
        "correlation" if stat_type == "pearson" else
        "f_stat"      if stat_type == "anova"   else
        "chi2"
    )

    rates   = sorted(all_rates_stats.keys())
    methods = list(next(iter(all_rates_stats.values())).keys())
    colors  = {
        "complete_case": "#9467BD",
        "mean_mode":     "#4C72B0",
        "knn":           "#DD8452",
        "mice":          "#55A868",
        "missforest":    "#C44E52",
        "pairwise":      "#8172B2"
    }

    fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharey=False)

    for ax, mech in zip(axes, ["MCAR", "MAR", "MNAR"]):
        for method in methods:
            y_vals = []
            for rate in rates:
                diffs = compute_stat_differences(
                    {mech: all_rates_stats[rate][method][mech]},
                    ground_truth_stats, stat_type, value_col
                )
                y_vals.append(diffs["abs_diff"].mean() if not diffs.empty else np.nan)

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

    save_path = os.path.join(save_dir, f"degradation_{stat_type}.png") if save_dir else None
    _save_or_show(save_path)


# ── Degradation — aggregated across datasets ─────────────────

def plot_degradation_aggregated(stat_type, all_rates_stats, ground_truth_stats, save_dir=None):
    """
    One subplot per imputation method, one line per mechanism.
    Aggregated (averaged) across all datasets.
    """
    from evaluation.metrics import compute_stat_differences

    value_col = (
        "correlation" if stat_type == "pearson" else
        "f_stat"      if stat_type == "anova"   else
        "chi2"
    )

    rates       = sorted(all_rates_stats.keys())
    methods     = [m for m in next(iter(all_rates_stats.values())).keys() if m != "pairwise"]
    mechs       = ["MCAR", "MAR", "MNAR"]
    mech_colors = {"MCAR": "#55A868", "MAR": "#DD8452", "MNAR": "#C44E52"}

    ncols     = 3
    nrows     = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 6 * nrows), sharey=False)
    axes_flat = axes.flatten() if len(methods) > 1 else [axes]

    for ax, method in zip(axes_flat, methods):
        for mech in mechs:
            y_vals = []
            for rate in rates:
                diffs = compute_stat_differences(
                    {mech: all_rates_stats[rate][method][mech]},
                    ground_truth_stats, stat_type, value_col
                )
                y_vals.append(diffs["abs_diff"].mean() if not diffs.empty else np.nan)

            ax.plot([r * 100 for r in rates], y_vals,
                    marker="o", label=mech, color=mech_colors[mech],
                    linewidth=2.5, markersize=9)

        ax.set_title(method.replace("_", " ").title(), fontsize=15, fontweight="bold")
        ax.set_xlabel("Missing Rate (%)", fontsize=12)
        ax.set_ylabel("Mean Absolute Difference", fontsize=12)
        ax.set_xticks([r * 100 for r in rates])
        ax.legend(title="Mechanism", fontsize=11)
        ax.grid(linestyle="--", alpha=0.5)

    for ax in axes_flat[len(methods):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{stat_type.upper()}: Degradation per Method — Aggregated Across Datasets",
        fontsize=18, fontweight="bold"
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"degradation_aggregated_{stat_type}.png") if save_dir else None
    _save_or_show(save_path)