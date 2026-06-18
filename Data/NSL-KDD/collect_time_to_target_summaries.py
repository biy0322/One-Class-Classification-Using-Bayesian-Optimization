import argparse
from pathlib import Path

import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]


def fmt_mean_sem(mean, sem, digits=2):
    if pd.isna(mean):
        return "Not reached"
    if pd.isna(sem):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("NSL-KDD/results_budget_grid10"))
    parser.add_argument("--targets", nargs="+", type=float, default=[0.92, 0.93, 0.94, 0.95])
    args = parser.parse_args()

    rows = []
    latex_rows = []
    for target in args.targets:
        path = args.root / f"nsl_kdd_ocsvm_time_to_target_{target:.3f}_summary.csv"
        summary = pd.read_csv(path)
        for method in METHODS:
            row = summary[summary["method"] == method].iloc[0]
            out = {
                "target": target,
                "method": method,
                "reach": f"{int(row['n_reached'])}/{int(row['n_repeats'])}",
                "configs_to_target": fmt_mean_sem(
                    row["configs_to_target_mean"],
                    row["configs_to_target_sem"],
                    digits=1,
                ),
                "time_to_target_sec": fmt_mean_sem(
                    row["time_to_target_sec_mean"],
                    row["time_to_target_sec_sem"],
                    digits=2,
                ),
                "total_time_sec": fmt_mean_sem(
                    row["total_time_sec_mean"],
                    row["total_time_sec_sem"],
                    digits=2,
                ),
            }
            rows.append(out)
            latex_rows.append(
                f"{target:.3f} & {method} & {out['reach']} & "
                f"{out['configs_to_target']} & {out['time_to_target_sec']} \\\\"
            )

    combined = pd.DataFrame(rows)
    combined_path = args.root / "nsl_kdd_ocsvm_time_to_target_multi_summary.csv"
    latex_path = args.root / "nsl_kdd_ocsvm_time_to_target_multi_latex_rows.txt"
    combined.to_csv(combined_path, index=False)
    latex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")

    print(f"Saved: {combined_path}")
    print(f"Saved: {latex_path}")
    print(combined.to_string(index=False))
    print("\nLaTeX rows:")
    print(latex_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
