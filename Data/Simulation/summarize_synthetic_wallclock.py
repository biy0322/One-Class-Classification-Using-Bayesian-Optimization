import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]
INIT_CONFIGS = {
    "GC": 0,
    "RC": 0,
    "BO": 5,
    "HB": 0,
    "BOHB": 10,
}


def mean_sem(values):
    values = pd.Series(values, dtype=float)
    return values.mean(), values.sem(ddof=1)


def fmt_mean_sem(mean, sem, digits=2):
    if pd.isna(mean):
        return "Not reached"
    if pd.isna(sem):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def full_eval_mask(history):
    if "full_eval" not in history.columns:
        return pd.Series(True, index=history.index)
    return history["full_eval"].astype(str).str.lower().isin(["true", "1", "yes"])


def summarize_relative_targets(history, targets):
    history = history[full_eval_mask(history)].copy()
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
                configs_to_target = first["eval_index"] if pd.isna(trial_number) else int(trial_number) + 1
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


def summarize_targets_by_method(per_repeat):
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


def summarize_efficiency(time_to_best):
    rows = []
    for method in METHODS:
        group = time_to_best[time_to_best["method"] == method]
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
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sem"] = group[col].sem()
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean, sem, digits=2):
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def write_target_latex(summary, path):
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row['target_fraction']:.3f} & {row['method']} & "
            f"{int(row['n_reached'])}/{int(row['n_repeats'])} & "
            f"{fmt_mean_sem(row['configs_to_target_mean'], row['configs_to_target_sem'], 1)} & "
            f"{fmt_mean_sem(row['time_to_target_sec_mean'], row['time_to_target_sec_sem'], 2)} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_efficiency_latex(summary, path):
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
        "--result-dir",
        type=Path,
        default=Path("Simulation/results_synthetic_wallclock_95_5_sep2"),
    )
    parser.add_argument("--targets", nargs="+", type=float, default=[0.98, 0.99, 0.995])
    args = parser.parse_args()

    history = pd.read_csv(args.result_dir / "synthetic_ocsvm_search_history.csv")
    time_to_best = pd.read_csv(args.result_dir / "synthetic_ocsvm_time_to_best.csv")

    per_repeat = summarize_relative_targets(history, args.targets)
    target_summary = summarize_targets_by_method(per_repeat)
    efficiency_summary = summarize_efficiency(time_to_best)

    per_repeat.to_csv(args.result_dir / "synthetic_ocsvm_time_to_relative_target.csv", index=False)
    target_summary.to_csv(
        args.result_dir / "synthetic_ocsvm_time_to_relative_target_summary.csv",
        index=False,
    )
    efficiency_summary.to_csv(args.result_dir / "synthetic_ocsvm_efficiency_summary.csv", index=False)

    write_target_latex(
        target_summary,
        args.result_dir / "synthetic_ocsvm_time_to_relative_target_latex_rows.txt",
    )
    write_efficiency_latex(
        efficiency_summary,
        args.result_dir / "synthetic_ocsvm_efficiency_latex_rows.txt",
    )

    print("\nTime-to-relative-target summary")
    print(target_summary.to_string(index=False))
    print("\nComputational-efficiency summary")
    print(efficiency_summary.to_string(index=False))


if __name__ == "__main__":
    main()
