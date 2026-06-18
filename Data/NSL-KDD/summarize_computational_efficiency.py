import argparse
from pathlib import Path

import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]
INIT_CONFIGS = {
    "GC": 0,
    "RC": 0,
    "BO": 5,
    "HB": 0,
    "BOHB": 10,
}


def fmt(mean, sem, digits=2):
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def summarize(path):
    df = pd.read_csv(path)
    rows = []
    for method in METHODS:
        g = df[df["method"] == method]
        row = {"method": method, "initial_configurations": INIT_CONFIGS[method]}
        for col in [
            "total_configurations",
            "full_validation_evals",
            "total_observed_evals",
            "total_time_sec",
            "total_evaluation_time_sec",
            "optimization_overhead_sec",
            "optimization_overhead_pct",
        ]:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_sem"] = g[col].sem()
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex(summary, path):
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row['method']} & "
            f"{int(row['initial_configurations'])} & "
            f"{fmt(row['total_configurations_mean'], row['total_configurations_sem'], 1)} & "
            f"{fmt(row['full_validation_evals_mean'], row['full_validation_evals_sem'], 1)} & "
            f"{fmt(row['total_observed_evals_mean'], row['total_observed_evals_sem'], 1)} & "
            f"{fmt(row['total_time_sec_mean'], row['total_time_sec_sem'], 2)} & "
            f"{fmt(row['optimization_overhead_sec_mean'], row['optimization_overhead_sec_sem'], 2)} & "
            f"{fmt(row['optimization_overhead_pct_mean'], row['optimization_overhead_pct_sem'], 2)}\\% \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("NSL-KDD/results_efficiency_grid10/nsl_kdd_ocsvm_time_to_best.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("NSL-KDD/results_efficiency_grid10"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(args.results)
    summary_path = args.output_dir / "nsl_kdd_ocsvm_efficiency_summary.csv"
    latex_path = args.output_dir / "nsl_kdd_ocsvm_efficiency_latex_rows.txt"
    summary.to_csv(summary_path, index=False)
    write_latex(summary, latex_path)
    print(summary)
    print(f"Saved: {summary_path}")
    print(f"Saved: {latex_path}")


if __name__ == "__main__":
    main()
