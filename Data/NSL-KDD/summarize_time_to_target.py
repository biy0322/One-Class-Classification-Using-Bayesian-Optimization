import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]


def mean_sem(values):
    values = pd.Series(values, dtype=float)
    return values.mean(), values.sem(ddof=1)


def summarize_time_to_target(history, target):
    rows = []
    for (repeat, method), group in history.groupby(["repeat", "method"], sort=False):
        group = group.sort_values("elapsed_sec").copy()
        if "full_eval" in group.columns:
            group = group[group["full_eval"].astype(bool)]

        reached_group = group[group["best_so_far"] >= target]
        total_time = group["elapsed_sec"].max()
        total_observed_evals = group["eval_index"].max()

        if reached_group.empty:
            rows.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "target": target,
                    "reached": False,
                    "configs_to_target": np.nan,
                    "observed_evals_to_target": np.nan,
                    "time_to_target_sec": np.nan,
                    "total_time_sec": total_time,
                    "total_observed_evals": total_observed_evals,
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
                "target": target,
                "reached": True,
                "configs_to_target": configs_to_target,
                "observed_evals_to_target": first["eval_index"],
                "time_to_target_sec": first["elapsed_sec"],
                "total_time_sec": total_time,
                "total_observed_evals": total_observed_evals,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_method(per_repeat):
    rows = []
    for method in METHODS:
        group = per_repeat[per_repeat["method"] == method]
        reached = group[group["reached"]]
        row = {
            "method": method,
            "target": group["target"].iloc[0] if len(group) else np.nan,
            "n_repeats": int(len(group)),
            "n_reached": int(group["reached"].sum()),
            "reach_rate": float(group["reached"].mean()) if len(group) else np.nan,
        }

        for col in [
            "configs_to_target",
            "observed_evals_to_target",
            "time_to_target_sec",
            "total_time_sec",
        ]:
            if col == "total_time_sec":
                values = group[col]
            else:
                values = reached[col]
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
    parser.add_argument("--target", type=float, default=0.935)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("NSL-KDD/results_budget_grid10"),
    )
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    per_repeat = summarize_time_to_target(history, args.target)
    summary = summarize_by_method(per_repeat)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_repeat_path = args.output_dir / f"nsl_kdd_ocsvm_time_to_target_{args.target:.3f}.csv"
    summary_path = args.output_dir / f"nsl_kdd_ocsvm_time_to_target_{args.target:.3f}_summary.csv"
    per_repeat.to_csv(per_repeat_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {per_repeat_path}")
    print(f"Saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
