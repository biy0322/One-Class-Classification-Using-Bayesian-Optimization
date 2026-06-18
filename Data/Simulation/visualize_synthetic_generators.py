#!/usr/bin/env python
# coding: utf-8

"""Visualize original and overlap synthetic data generators."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_dataset import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument("--normal-std", type=float, default=0.5)
    parser.add_argument("--anomaly-std", type=float, default=0.6)
    parser.add_argument("--separations", nargs="+", type=float, default=[2.0, 1.5, 1.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Review_Overlap_RatioSep" / "synthetic_generator_comparison.png",
    )
    return parser.parse_args()


def combine_train_test(x_train, x_test, y_train, y_test):
    return np.r_[x_train, x_test], np.r_[y_train, y_test]


def plot_dataset(ax, x, y, title):
    normal = y == 0
    anomaly = y == 1
    ax.scatter(
        x[normal, 0],
        x[normal, 1],
        s=12,
        alpha=0.42,
        c="#4c78a8",
        edgecolors="none",
        label="Normal",
    )
    ax.scatter(
        x[anomaly, 0],
        x[anomaly, 1],
        s=26,
        alpha=0.84,
        c="#e45756",
        edgecolors="#6b1f1a",
        linewidths=0.25,
        label="Anomaly",
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.3)


def main():
    args = parse_args()
    dataset = Dataset()

    panels = []
    x_train, x_test, y_train, y_test = dataset.generate_data(
        n_samples=args.n_samples,
        outlier_fraction=args.outlier_fraction,
        random_state=args.seed,
    )
    x, y = combine_train_test(x_train, x_test, y_train, y_test)
    panels.append(("Original generator\n(no separation parameter)", x, y))

    for separation in args.separations:
        x_train, x_test, y_train, y_test = dataset.generate_overlap_data(
            n_samples=args.n_samples,
            outlier_fraction=args.outlier_fraction,
            separation=separation,
            normal_std=args.normal_std,
            anomaly_std=args.anomaly_std,
            random_state=args.seed,
        )
        x, y = combine_train_test(x_train, x_test, y_train, y_test)
        panels.append((f"Overlap generator\nseparation = {separation:g}", x, y))

    if len(panels) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0), sharex=True, sharey=True)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.15 * len(panels), 4.2), sharex=True, sharey=True)
        if len(panels) == 1:
            axes = [axes]

    for ax, (title, x, y) in zip(axes, panels):
        plot_dataset(ax, x, y, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    ratio = f"{int(round((1 - args.outlier_fraction) * 100))}:{int(round(args.outlier_fraction * 100))}"
    fig.suptitle(f"Synthetic Data Generation Comparison (normal:anomaly = {ratio})", y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
