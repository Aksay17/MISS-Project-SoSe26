# ============================================================
# visualization/plots.py
# All plotting functions.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOTS_DIR


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

def plot_rank(stat_type, all_rates_stats, ground_truth_stats):
    """
    Aggregated average-rank plot, pooled across ALL missingness rates.

    Within every (rate, mechanism, dataset) situation the methods are ranked by
    their mean absolute difference from ground truth (rank 1 = closest = best),
    and each method's ranks are then averaged. The 'Overall' panel averages over
    all rates, mechanisms and datasets; each mechanism panel averages over all
    rates and datasets for that mechanism.
    """
    from evaluation.metrics import compute_stat_differences

    value_col = (
        "correlation" if stat_type == "pearson" else
        "f_stat"      if stat_type == "anova"   else
        "chi2"
    )

    # One long table of per-pair differences across every rate and method.
    all_data = []
    for rate, all_stats in all_rates_stats.items():
        for method_name, stats_dict in all_stats.items():
            df = compute_stat_differences(stats_dict, ground_truth_stats,
                                          stat_type, value_col)
            if df.empty:
                continue
            df = df.copy()
            df["method"] = method_name
            df["rate"]   = rate
            all_data.append(df)

    if not all_data:
        print(f"No data for {stat_type}")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Mean score per situation, then rank methods within each
    # (rate, mechanism, dataset) before averaging the ranks.
    grouped = (combined
               .groupby(["rate", "mechanism", "dataset", "method"])["abs_diff"]
               .mean().reset_index())
    grouped["rank"] = grouped.groupby(["rate", "mechanism", "dataset"])["abs_diff"].rank(
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
    ax.set_xlim(0, 6.5)
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
        ax.set_xlim(0, 6.5)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, mech_df["rank"]):
            if pd.isna(val):
                continue
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=12)

    fig.suptitle(f"{stat_type.upper()}: Method Rankings Aggregated Across All Missingness Rates",
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


# ── Cell-level imputation recovery plots ─────────────────────

def plot_recovery_heatmap(accuracy_df, metric, rate=None):
    """
    Cell-level recovery heatmap (rows = datasets, cols = methods), one per
    mechanism, at a single missing rate (default = the lowest rate present).
        metric="nrmse"        -> lower = better  (red colormap)
        metric="cat_accuracy" -> higher = better (green colormap, fixed 0..1)
    """
    if accuracy_df.empty:
        print("No recovery data")
        return
    if rate is None:
        rate = min(accuracy_df["rate"].unique())

    df = accuracy_df[accuracy_df["rate"] == rate]
    if metric == "nrmse":
        cmap, label, vmin, vmax = "YlOrRd", "NRMSE (lower = better)", None, None
    else:
        cmap, label, vmin, vmax = "YlGn", "Categorical Accuracy (higher = better)", 0.0, 1.0

    for mech in ["MCAR", "MAR", "MNAR"]:
        mech_df = df[df["mechanism"] == mech].pivot(
            index="dataset", columns="method", values=metric
        )
        if mech_df.empty:
            continue

        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(mech_df, annot=True, fmt=".3f", cmap=cmap,
                         vmin=vmin, vmax=vmax, linewidths=0.5,
                         cbar_kws={"label": label})
        ax.set_facecolor("#eeeeee")
        for yi in range(mech_df.shape[0]):
            for xi in range(mech_df.shape[1]):
                if pd.isna(mech_df.iloc[yi, xi]):
                    ax.text(xi + 0.5, yi + 0.5, "N/A", ha="center", va="center",
                            fontsize=11, color="grey", fontweight="bold")

        plt.title(f"Imputation Recovery ({metric}) - {mech} @ {int(rate*100)}% missing",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Imputation Method", fontsize=12)
        plt.ylabel("Cancer Dataset", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, f"heatmap_recovery_{metric}_{mech}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()


def plot_recovery_degradation(accuracy_df, metric):
    """
    Cell-level recovery vs missing rate, one line per method, per-mechanism panels.
        metric="nrmse"        -> expect lines to rise (recovery worsens)
        metric="cat_accuracy" -> expect lines to fall
    """
    if accuracy_df.empty:
        print("No recovery data")
        return

    rates   = sorted(accuracy_df["rate"].unique())
    methods = list(accuracy_df["method"].unique())
    colors  = {"mean_mode": "#4C72B0", "knn": "#DD8452",
               "mice": "#55A868", "missforest": "#C44E52"}
    ylabel  = ("NRMSE (lower = better)" if metric == "nrmse"
               else "Categorical Accuracy (higher = better)")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, mech in zip(axes, ["MCAR", "MAR", "MNAR"]):
        sub = accuracy_df[accuracy_df["mechanism"] == mech]
        for method in methods:
            g = sub[sub["method"] == method].groupby("rate")[metric].mean().reindex(rates)
            ax.plot([r * 100 for r in rates], g.values, marker="o", label=method,
                    color=colors.get(method, "gray"), linewidth=2, markersize=7)
        ax.set_title(mech, fontsize=16, fontweight="bold")
        ax.set_xlabel("Missing Rate (%)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks([r * 100 for r in rates])
        if metric == "cat_accuracy":
            ax.set_ylim(-0.02, 1.02)
        ax.legend(title="Method", fontsize=11)
        ax.grid(linestyle="--", alpha=0.5)

    fig.suptitle(f"Imputation Value Recovery ({metric}) by Missingness Rate",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"degradation_recovery_{metric}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_recovery_bar(accuracy_df, metric):
    """
    Grouped bar chart of cell-level recovery across ALL missing rates.
    One panel per mechanism (MCAR/MAR/MNAR); within each panel x = imputation
    method and bars are grouped by missing rate. y = mean metric across datasets.
        metric="nrmse"        -> lower = better
        metric="cat_accuracy" -> higher = better (y fixed 0..1)
    """
    if accuracy_df.empty:
        print("No recovery data")
        return

    order   = ["mean_mode", "knn", "mice", "missforest"]
    methods = [m for m in order if m in accuracy_df["method"].unique()]
    mechs   = ["MCAR", "MAR", "MNAR"]
    rates   = sorted(accuracy_df["rate"].unique())
    ylabel  = ("NRMSE (lower = better)" if metric == "nrmse"
               else "Categorical Accuracy (higher = better)")
    rate_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(rates), 1)))

    x     = np.arange(len(methods))
    width = 0.8 / max(len(rates), 1)

    fig, axes = plt.subplots(1, 3, figsize=(26, 7), sharey=True)
    for ax, mech in zip(axes, mechs):
        sub   = accuracy_df[accuracy_df["mechanism"] == mech]
        table = (sub.groupby(["method", "rate"])[metric].mean()
                    .unstack("rate").reindex(index=methods, columns=rates))
        for i, rate in enumerate(rates):
            vals   = table[rate].to_numpy()
            offset = (i - (len(rates) - 1) / 2) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=f"{int(rate*100)}%", color=rate_colors[i], alpha=0.9)
            for b, v in zip(bars, vals):
                if pd.isna(v):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_title(mech, fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20)
        ax.set_xlabel("Imputation Method", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.legend(title="Missing Rate", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if metric == "cat_accuracy":
            ax.set_ylim(0, 1.05)

    fig.suptitle(f"Imputation Recovery ({metric}) by Method and Rate",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"recovery_bar_{metric}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()