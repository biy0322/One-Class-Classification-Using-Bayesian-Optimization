import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]


def mean_sem(values):
    values = pd.Series(values, dtype=float)
    return values.mean(), values.sem(ddof=1)


def fmt_mean_sem(mean, sem, digits=2):
    if pd.isna(mean):
        return "Not reached"
    if pd.isna(sem):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def summarize_relative_targets(history, targets):
    if "full_eval" in history.columns:
        history = history[history["full_eval"].astype(bool)]

    rows = []
    for repeat, rep_group in history.groupby("repeat", sort=False):
        split_best = rep_group["score"].max()
        for target_frac in targets:
            target_score = target_frac * split_best
            for method in METHODS:
                group = rep_group[rep_group["method"] == method].sort_values("elapsed_sec")
                reached_group = group[group["best_so_far"] >= target_score]
                total_time = group["elapsed_sec"].max()
                if reached_group.empty:
                    rows.append(
                        {
                            "repeat": repeat,
                            "method": method,
                            "target_fraction": target_frac,
                            "split_best_validation_f1": split_best,
                            "target_validation_f1": target_score,
                            "reached": False,
                            "configs_to_target": np.nan,
                            "time_to_target_sec": np.nan,
                            "total_time_sec": total_time,
                        }
                    )
                    continue

                first = reached_group.iloc[0]
                trial_number = first.get("trial_number")
                if pd.isna(trial_number):
                    configs_to_target = first["eval_index"]
                else:
                    configs_to_target = int(trial_number) + 1
                rows.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "target_fraction": target_frac,
                        "split_best_validation_f1": split_best,
                        "target_validation_f1": target_score,
                        "reached": True,
                        "configs_to_target": configs_to_target,
                        "time_to_target_sec": first["elapsed_sec"],
                        "total_time_sec": total_time,
                    }
                )
    return pd.DataFrame(rows)


def summarize_by_method(per_repeat):
    rows = []
    for target_fraction in sorted(per_repeat["target_fraction"].unique()):
        frac_group = per_repeat[per_repeat["target_fraction"] == target_fraction]
        for method in METHODS:
            group = frac_group[frac_group["method"] == method]
            reached = group[group["reached"]]
            row = {
                "target_fraction": target_fraction,
                "method": method,
                "n_repeats": int(len(group)),
                "n_reached": int(group["reached"].sum()),
                "reach_rate": float(group["reached"].mean()) if len(group) else np.nan,
            }
            for col in ["target_validation_f1", "configs_to_target", "time_to_target_sec"]:
                values = reached[col] if col != "target_validation_f1" else group[col]
                mean, sem = mean_sem(values)
                row[f"{col}_mean"] = mean
                row[f"{col}_sem"] = sem
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("NSL-KDD/results_budget_grid10/nsl_kdd_ocsvm_search_history.csv"),
    )
    parser.add_argument("--targets", nargs="+", type=float, default=[0.98, 0.99, 0.995])
    parser.add_argument("--output-dir", type=Path, default=Path("NSL-KDD/results_budget_grid10"))
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    per_repeat = summarize_relative_targets(history, args.targets)
    summary = summarize_by_method(per_repeat)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_repeat_path = args.output_dir / "nsl_kdd_ocsvm_time_to_relative_target.csv"
    summary_path = args.output_dir / "nsl_kdd_ocsvm_time_to_relative_target_summary.csv"
    latex_path = args.output_dir / "nsl_kdd_ocsvm_time_to_relative_target_latex_rows.txt"
    per_repeat.to_csv(per_repeat_path, index=False)
    summary.to_csv(summary_path, index=False)

    latex_rows = []
    for _, row in summary.iterrows():
        latex_rows.append(
            f"{row['target_fraction']:.3f} & {row['method']} & "
            f"{int(row['n_reached'])}/{int(row['n_repeats'])} & "
            f"{fmt_mean_sem(row['configs_to_target_mean'], row['configs_to_target_sem'], 1)} & "
            f"{fmt_mean_sem(row['time_to_target_sec_mean'], row['time_to_target_sec_sem'], 2)} \\\\"
        )
    latex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")

    print(f"Saved: {per_repeat_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {latex_path}")
    print(summary.to_string(index=False))
    print("\nLaTeX rows:")
    print(latex_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
