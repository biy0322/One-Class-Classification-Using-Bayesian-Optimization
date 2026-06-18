#!/usr/bin/env python
# coding: utf-8

"""Plot ratio-by-overlap results for the review experiment."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["grid", "random", "bayesian", "hyperband", "bohb"]
METHOD_LABELS = {
    "grid": "Grid",
    "random": "Random",
    "bayesian": "Bayesian",
    "hyperband": "Hyperband",
    "bohb": "BOHB",
}
COLORS = {
    "grid": "#4c78a8",
    "random": "#f58518",
    "bayesian": "#54a24b",
    "hyperband": "#b279a2",
    "bohb": "#e45756",
}
MARKERS = {
    "grid": "o",
    "random": "s",
    "bayesian": "^",
    "hyperband": "D",
    "bohb": "P",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "Review_Overlap_RatioSep",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--budget", type=Path, default=None)
    return parser.parse_args()


def ratio_labels(df):
    ratios = sorted(df["outlier_fraction"].unique())
    labels = []
    for ratio in ratios:
        subset = df[df["outlier_fraction"] == ratio]
        if "normal_to_anomaly_ratio" in subset.columns:
            labels.append(str(subset["normal_to_anomaly_ratio"].iloc[0]))
        else:
            labels.append(f"{int(round((1 - ratio) * 100))}:{int(round(ratio * 100))}")
    return ratios, labels


def separation_order(df):
    return sorted(df["separation"].unique(), reverse=True)


def plot_metric(df, metric, ylabel, output_path):
    metric_df = df[df["metric"] == metric].copy()
    ratios, labels = ratio_labels(metric_df)
    separations = separation_order(metric_df)

    fig, axes = plt.subplots(1, len(separations), figsize=(15.2, 4.6), sharey=True)
    if len(separations) == 1:
        axes = [axes]

    for ax, separation in zip(axes, separations):
        sep_df = metric_df[metric_df["separation"] == separation]
        for method in METHOD_ORDER:
            method_df = sep_df[sep_df["method"] == method].sort_values("outlier_fraction")
            ax.errorbar(
                method_df["outlier_fraction"],
                method_df["mean"],
                yerr=method_df["sem"],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.8,
                markersize=5.8,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"Separation = {separation:g}")
        ax.set_xlabel("Normal:Anomaly ratio")
        ax.set_xticks(ratios)
        ax.set_xticklabels(labels, rotation=25)
        ax.grid(True, linewidth=0.45, alpha=0.35)

    axes[0].set_ylabel(ylabel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(ylabel, y=0.98, fontsize=15)
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_budget(df, value_col, sem_col, ylabel, title, output_path):
    ratios, labels = ratio_labels(df)
    separations = separation_order(df)

    fig, axes = plt.subplots(1, len(separations), figsize=(15.2, 4.6), sharey=True)
    if len(separations) == 1:
        axes = [axes]

    for ax, separation in zip(axes, separations):
        sep_df = df[df["separation"] == separation]
        for method in METHOD_ORDER:
            method_df = sep_df[sep_df["method"] == method].sort_values("outlier_fraction")
            ax.errorbar(
                method_df["outlier_fraction"],
                method_df[value_col],
                yerr=method_df[sem_col],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.8,
                markersize=5.8,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"Separation = {separation:g}")
        ax.set_xlabel("Normal:Anomaly ratio")
        ax.set_xticks(ratios)
        ax.set_xticklabels(labels, rotation=25)
        ax.grid(True, linewidth=0.45, alpha=0.35)

    axes[0].set_ylabel(ylabel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(title, y=0.98, fontsize=15)
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def main():
    args = parse_args()
    root = args.root
    summary_path = args.summary or root / "overlap_summary_with_time.csv"
    budget_path = args.budget or root / "budget_to_best_summary.csv"

    summary = pd.read_csv(summary_path)
    budget = pd.read_csv(budget_path)

    plot_metric(
        summary,
        "f1",
        "F1-score",
        root / "f1_by_ratio_separation.png",
    )
    plot_metric(
        summary,
        "roc_auc",
        "AUC",
        root / "auc_by_ratio_separation.png",
    )
    plot_budget(
        budget,
        "trials_to_best_mean",
        "trials_to_best_sem",
        "Trials to first best score",
        "Budget to First Best Score",
        root / "budget_to_best_trials_by_ratio_separation.png",
    )
    plot_budget(
        budget,
        "time_to_best_sec_mean",
        "time_to_best_sec_sem",
        "Seconds to first best score",
        "Time to First Best Score",
        root / "budget_to_best_time_by_ratio_separation.png",
    )


if __name__ == "__main__":
    main()
