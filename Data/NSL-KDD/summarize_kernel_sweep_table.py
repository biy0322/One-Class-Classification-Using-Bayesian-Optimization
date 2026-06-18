import argparse
from pathlib import Path

import pandas as pd


KERNEL_LABELS = {
    "rbf": "RBF (squared exponential)",
    "matern": "Matern",
    "rational_quadratic": "Rational quadratic",
}
KERNEL_ORDER = ["rbf", "matern", "rational_quadratic"]


def fmt(mean, sem, digits=4):
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("NSL-KDD/results_sensitivity_binary_f1/nsl_kdd_kernel_sweep_f1_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NSL-KDD/results_sensitivity_binary_f1/nsl_kdd_kernel_sweep_f1_table.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["bo_kernel"].isin(KERNEL_ORDER)].copy()
    df["kernel_label"] = df["bo_kernel"].map(KERNEL_LABELS)
    df["bo_kernel"] = pd.Categorical(df["bo_kernel"], categories=KERNEL_ORDER, ordered=True)
    df = df.sort_values("bo_kernel")

    table = pd.DataFrame(
        {
            "Kernel": df["kernel_label"],
            "Recall": [fmt(m, s) for m, s in zip(df["Recall_mean"], df["Recall_sem"])],
            "F-1": [fmt(m, s) for m, s in zip(df["F-1_mean"], df["F-1_sem"])],
            "AUC": [fmt(m, s) for m, s in zip(df["AUC_mean"], df["AUC_sem"])],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    latex_path = args.output.with_name(args.output.stem + "_latex_rows.txt")
    lines = [
        f"{row['Kernel']} & {row['Recall']} & {row['F-1']} & {row['AUC']} \\\\"
        for _, row in table.iterrows()
    ]
    latex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {args.output}")
    print(f"Saved: {latex_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
