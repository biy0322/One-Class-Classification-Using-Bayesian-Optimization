#!/usr/bin/env python
# coding: utf-8

"""Visualize synthetic overlap strength by separation value."""

from pathlib import Path
import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_dataset import Dataset


def load_or_generate(args, separation):
    scenario = (
        f"overlap_ratio_{args.outlier_fraction:.2f}_"
        f"sep_{separation:.2f}_"
        f"nstd_{args.normal_std:.2f}_"
        f"astd_{args.anomaly_std:.2f}"
    ).replace(".", "p")
    data_dir = args.review_root / scenario / "Data"

    paths = {
        "x_train": data_dir / "X_train_dataset.npy",
        "y_train": data_dir / "y_train_dataset.npy",
        "x_test": data_dir / "X_test_dataset.npy",
        "y_test": data_dir / "y_test_dataset.npy",
    }
    if all(path.exists() for path in paths.values()):
        x_train = np.load(paths["x_train"], allow_pickle=True)[args.repeat_index]
        y_train = np.load(paths["y_train"], allow_pickle=True)[args.repeat_index]
        x_test = np.load(paths["x_test"], allow_pickle=True)[args.repeat_index]
        y_test = np.load(paths["y_test"], allow_pickle=True)[args.repeat_index]
        return np.r_[x_train, x_test], np.r_[y_train, y_test], "saved"

    dataset = Dataset()
    x_train, x_test, y_train, y_test = dataset.generate_overlap_data(
        n_samples=args.n_samples,
        outlier_fraction=args.outlier_fraction,
        separation=separation,
        normal_std=args.normal_std,
        anomaly_std=args.anomaly_std,
        random_state=args.seed + args.repeat_index,
    )
    return np.r_[x_train, x_test], np.r_[y_train, y_test], "generated"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--separations", nargs="+", type=float, default=[2.0, 1.25, 0.75, 0.5])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--normal-std", type=float, default=0.5)
    parser.add_argument("--anomaly-std", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--review-root", type=Path, default=Path(__file__).resolve().parent / "Review_Overlap")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "Review_Overlap" / "overlap_strength_scatter.png")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = []
    for separation in args.separations:
        x, y, source = load_or_generate(args, separation)
        datasets.append((separation, x, y, source))

    all_x = np.vstack([x for _, x, _, _ in datasets])
    padding = 0.35
    x_min, y_min = all_x.min(axis=0) - padding
    x_max, y_max = all_x.max(axis=0) + padding

    n_cols = min(2, len(datasets))
    n_rows = math.ceil(len(datasets) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.8 * n_rows), squeeze=False)

    for ax, (separation, x, y, source) in zip(axes.ravel(), datasets):
        normal = y == 0
        anomaly = y == 1
        ax.scatter(x[normal, 0], x[normal, 1], s=12, alpha=0.45, c="#377eb8", label="Normal", edgecolors="none")
        ax.scatter(x[anomaly, 0], x[anomaly, 1], s=28, alpha=0.80, c="#e41a1c", label="Anomaly", edgecolors="white", linewidths=0.35)
        ax.set_title(f"separation = {separation:g}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.45, alpha=0.35)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.text(
            0.02,
            0.98,
            f"normal={normal.sum()}, anomaly={anomaly.sum()}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )

    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Synthetic OCC Overlap Strength by Separation", y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
