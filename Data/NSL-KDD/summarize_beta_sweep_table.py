import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["GC", "RC", "BO", "HB", "BOHB"]
BETA_ORDER = [0.5, 1.0, 2.0, 3.0]
METRICS = ["Recall", "F-1", "AUC"]


def fmt(mean, sem, digits=4):
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def fmt_latex(mean, sem, is_best, digits=4):
    value = fmt(mean, sem, digits)
    if is_best:
        return rf"\textbf{{{value}}}"
    return value


def beta_label(beta):
    return f"{beta:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("NSL-KDD/results_sensitivity_binary_f1/nsl_kdd_sensitivity_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NSL-KDD/results_sensitivity_binary_f1/nsl_kdd_beta_sweep_25_table.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["experiment_group"].eq("beta_sweep")].copy()
    df = df[df["validation_average"].eq("binary")].copy()
    df["validation_beta"] = df["validation_beta"].astype(float)
    df = df[df["validation_beta"].isin(BETA_ORDER)].copy()
    df["validation_beta"] = pd.Categorical(
        df["validation_beta"], categories=BETA_ORDER, ordered=True
    )
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["validation_beta", "method"])

    table_rows = []
    latex_lines = []
    for beta, group in df.groupby("validation_beta", observed=True):
        best_by_metric = {
            metric: group[f"{metric}_mean"].max()
            for metric in METRICS
        }
        for _, row in group.iterrows():
            out = {
                "Validation beta": beta_label(float(beta)),
                "Method": row["method"],
            }
            latex_values = []
            for metric in METRICS:
                mean = row[f"{metric}_mean"]
                sem = row[f"{metric}_sem"]
                is_best = mean == best_by_metric[metric]
                out[metric] = fmt(mean, sem)
                latex_values.append(fmt_latex(mean, sem, is_best))
            table_rows.append(out)
            latex_lines.append(
                f"{out['Validation beta']} & {out['Method']} & "
                f"{latex_values[0]} & {latex_values[1]} & {latex_values[2]} \\\\"
            )

    table = pd.DataFrame(table_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    latex_path = args.output.with_name(args.output.stem + "_latex_rows.txt")
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    print(f"Saved: {args.output}")
    print(f"Saved: {latex_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
