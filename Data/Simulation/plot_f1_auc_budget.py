#!/usr/bin/env python
# coding: utf-8

"""Plot F1/AUC only and summarize search budgets."""

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
    parser.add_argument("--separations", nargs="+", type=float, default=[2.0, 1.25])
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "Review_Overlap" / "sep_2p00_1p25",
    )
    return parser.parse_args()


def plot_metric(ax, df, metric, title, ylabel):
    metric_df = df[df["metric"] == metric]
    for method in METHOD_ORDER:
        method_df = metric_df[metric_df["method"] == method].sort_values("separation", ascending=False)
        ax.errorbar(
            method_df["separation"],
            method_df["mean"],
            yerr=method_df["sem"],
            marker=MARKERS[method],
            color=COLORS[method],
            linewidth=1.9,
            markersize=6,
            capsize=3,
            label=METHOD_LABELS[method],
        )
    ax.set_title(title)
    ax.set_xlabel("Separation")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(df["separation"].unique(), reverse=True))
    ax.invert_xaxis()
    ax.grid(True, linewidth=0.45, alpha=0.35)


def build_budget_table(df):
    base = df[df["metric"] == "f1"].copy()
    rows = []
    for separation, sep_df in base.groupby("separation"):
        for _, row in sep_df.iterrows():
            n_timing_rows = int(round(row["time_total_sec"] / row["time_mean_sec"]))
            rows.append({
                "separation": separation,
                "method": row["method"],
                "method_label": METHOD_LABELS[row["method"]],
                "budget_per_fold": int(row["budget_evaluations"]),
                "n_folds_total": n_timing_rows,
                "total_budget": int(row["budget_evaluations"] * n_timing_rows),
                "mean_search_time_sec": row["time_mean_sec"],
            })
    return pd.DataFrame(rows).sort_values(["separation", "method"], ascending=[False, True])


def plot_budget(budget_df, output_path):
    methods = METHOD_ORDER
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    values = [
        budget_df[budget_df["method"] == method]["budget_per_fold"].iloc[0]
        for method in methods
    ]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [COLORS[method] for method in methods]

    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Search evaluations per fold")
    ax.set_title("Hyperparameter Search Budget")
    ax.grid(axis="y", linewidth=0.45, alpha=0.35)
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def main():
    args = parse_args()
    df = pd.read_csv(args.summary)
    df = df[df["separation"].isin(args.separations)].copy()
    if df.empty:
        raise ValueError("No rows matched requested separation values.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    plot_metric(axes[0], df, "f1", "F1-score", "F1-score")
    plot_metric(axes[1], df, "roc_auc", "AUC", "AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("F1-score and AUC under Mild Overlap", y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])

    perf_png = args.output_prefix.with_name(args.output_prefix.name + "_f1_auc.png")
    fig.savefig(perf_png, dpi=220, bbox_inches="tight")
    fig.savefig(perf_png.with_suffix(".pdf"), bbox_inches="tight")

    budget_df = build_budget_table(df)
    budget_csv = args.output_prefix.with_name(args.output_prefix.name + "_search_budget.csv")
    budget_df.to_csv(budget_csv, index=False)

    budget_png = args.output_prefix.with_name(args.output_prefix.name + "_search_budget.png")
    plot_budget(budget_df, budget_png)

    print(f"Saved: {perf_png}")
    print(f"Saved: {perf_png.with_suffix('.pdf')}")
    print(f"Saved: {budget_csv}")
    print(f"Saved: {budget_png}")
    print(f"Saved: {budget_png.with_suffix('.pdf')}")
    print(budget_df.to_string(index=False))


if __name__ == "__main__":
    main()
