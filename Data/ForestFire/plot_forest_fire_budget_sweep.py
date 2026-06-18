from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]


def main():
    out_dir = Path("results_budget_sweep")
    df = pd.read_csv(out_dir / "forest_fire_budget_sweep_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for method in METHODS:
        group = df[df["method"] == method].sort_values("budget")
        axes[0].errorbar(
            group["budget"],
            group["F-1_mean"],
            yerr=group["F-1_sem"],
            marker="o",
            label=method,
        )
        axes[1].errorbar(
            group["budget"],
            group["AUC_mean"],
            yerr=group["AUC_sem"],
            marker="o",
            label=method,
        )

    axes[0].set_ylabel("Final test F1")
    axes[1].set_ylabel("Final test AUC")
    for ax in axes:
        ax.set_xlabel("Configuration budget")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "forest_fire_budget_sweep_f1_auc.png", dpi=300)


if __name__ == "__main__":
    main()
