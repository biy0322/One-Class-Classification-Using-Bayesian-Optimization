#!/usr/bin/env python
# coding: utf-8

"""Plot overlap robustness results across separation values."""

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
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "Review_Overlap" / "overlap_summary_with_time.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Review_Overlap" / "overlap_performance_time.png",
    )
    parser.add_argument(
        "--separations",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of separation values to plot.",
    )
    return parser.parse_args()


def plot_metric(ax, df, metric, ylabel):
    metric_df = df[df["metric"] == metric]
    for method in METHOD_ORDER:
        method_df = metric_df[metric_df["method"] == method].sort_values("separation", ascending=False)
        ax.errorbar(
            method_df["separation"],
            method_df["mean"],
            yerr=method_df["sem"],
            marker=MARKERS[method],
            color=COLORS[method],
            linewidth=1.8,
            markersize=5.8,
            capsize=3,
            label=METHOD_LABELS[method],
        )
    ax.set_title(ylabel)
    ax.set_xlabel("Separation")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["separation"].unique(), reverse=True))
    ax.invert_xaxis()
    ax.grid(True, linewidth=0.45, alpha=0.35)


def plot_time(ax, df):
    time_df = df[df["metric"] == "f1"]
    for method in METHOD_ORDER:
        method_df = time_df[time_df["method"] == method].sort_values("separation", ascending=False)
        ax.errorbar(
            method_df["separation"],
            method_df["time_mean_sec"],
            yerr=method_df["time_sem_sec"],
            marker=MARKERS[method],
            color=COLORS[method],
            linewidth=1.8,
            markersize=5.8,
            capsize=3,
            label=METHOD_LABELS[method],
        )
    ax.set_title("Search Time")
    ax.set_xlabel("Separation")
    ax.set_ylabel("Mean search time per fold (sec)")
    ax.set_xticks(sorted(df["separation"].unique(), reverse=True))
    ax.invert_xaxis()
    ax.grid(True, linewidth=0.45, alpha=0.35)


def main():
    args = parse_args()
    df = pd.read_csv(args.summary)
    if args.separations is not None:
        df = df[df["separation"].isin(args.separations)].copy()
        if df.empty:
            raise ValueError("No rows matched the requested --separations values.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=False)
    plot_metric(axes[0, 0], df, "recall", "Recall")
    plot_metric(axes[0, 1], df, "f1", "F1-score")
    plot_metric(axes[1, 0], df, "roc_auc", "AUC")
    plot_time(axes[1, 1], df)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Overlap Robustness and Search Time", y=0.985, fontsize=15)
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"Saved: {pdf_output}")


if __name__ == "__main__":
    main()
