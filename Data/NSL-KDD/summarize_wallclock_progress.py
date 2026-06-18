import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]
METHOD_LABELS = {
    "GC": "Grid",
    "RC": "Random",
    "BO": "BO",
    "HB": "Hyperband",
    "BOHB": "BOHB",
}
COLORS = {
    "GC": "#4C78A8",
    "RC": "#F58518",
    "BO": "#54A24B",
    "HB": "#B279A2",
    "BOHB": "#E45756",
}


def mean_sem(values):
    values = pd.Series(values, dtype=float)
    return values.mean(), values.sem(ddof=1)


def fmt_mean_sem(mean, sem, digits=3):
    if pd.isna(mean):
        return "NA"
    if pd.isna(sem):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def full_eval_mask(history):
    if "full_eval" not in history.columns:
        return pd.Series(True, index=history.index)
    return history["full_eval"].astype(str).str.lower().isin(["true", "1", "yes"])


def value_at_time(group, time_sec, missing_value=np.nan):
    reached = group[group["elapsed_sec"] <= time_sec]
    if reached.empty:
        return missing_value
    return float(reached.iloc[-1]["relative_best_so_far"])


def build_full_history(history):
    full = history[full_eval_mask(history)].copy()
    full = full.sort_values(["repeat", "method", "elapsed_sec", "eval_index"])
    split_best = full.groupby("repeat")["score"].max().rename("split_best_validation_f1")
    full = full.merge(split_best, on="repeat", how="left")
    full["relative_score"] = full["score"] / full["split_best_validation_f1"]
    full["relative_best_so_far"] = full["best_so_far"] / full["split_best_validation_f1"]
    return full


def summarize_fixed_times(full, fixed_times):
    rows = []
    for repeat, rep_group in full.groupby("repeat", sort=False):
        split_best = float(rep_group["split_best_validation_f1"].iloc[0])
        for method in METHODS:
            group = rep_group[rep_group["method"] == method].sort_values("elapsed_sec")
            if group.empty:
                continue
            final_relative = float(group.iloc[-1]["relative_best_so_far"])
            total_time = float(group["elapsed_sec"].max())
            for time_sec in fixed_times:
                rows.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "time_sec": float(time_sec),
                        "split_best_validation_f1": split_best,
                        "relative_best_so_far": value_at_time(group, time_sec),
                        "final_relative_best_so_far": final_relative,
                        "method_total_time_sec": total_time,
                    }
                )
            rows.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "time_sec": "final",
                    "split_best_validation_f1": split_best,
                    "relative_best_so_far": final_relative,
                    "final_relative_best_so_far": final_relative,
                    "method_total_time_sec": total_time,
                }
            )
    return pd.DataFrame(rows)


def summarize_fixed_by_method(snapshot):
    rows = []
    ordered_times = [t for t in snapshot["time_sec"].drop_duplicates()]
    ordered_times = sorted([t for t in ordered_times if t != "final"], key=float) + ["final"]
    for time_sec in ordered_times:
        time_group = snapshot[snapshot["time_sec"] == time_sec]
        for method in METHODS:
            group = time_group[time_group["method"] == method]
            values = group["relative_best_so_far"].dropna()
            mean, sem = mean_sem(values)
            rows.append(
                {
                    "time_sec": time_sec,
                    "method": method,
                    "n_repeats": int(len(group)),
                    "n_available": int(values.shape[0]),
                    "relative_best_so_far_mean": mean,
                    "relative_best_so_far_sem": sem,
                    "pct_of_target_mean": 100.0 * mean if pd.notna(mean) else np.nan,
                    "pct_of_target_sem": 100.0 * sem if pd.notna(sem) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_curve(full, time_grid):
    rows = []
    for repeat, rep_group in full.groupby("repeat", sort=False):
        for method in METHODS:
            group = rep_group[rep_group["method"] == method].sort_values("elapsed_sec")
            if group.empty:
                continue
            for time_sec in time_grid:
                rows.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "time_sec": float(time_sec),
                        "relative_best_so_far": value_at_time(group, time_sec, missing_value=0.0),
                    }
                )
    return pd.DataFrame(rows)


def summarize_curve(curve):
    rows = []
    for (method, time_sec), group in curve.groupby(["method", "time_sec"], sort=False):
        values = group["relative_best_so_far"].dropna()
        mean, sem = mean_sem(values)
        rows.append(
            {
                "method": method,
                "time_sec": float(time_sec),
                "n_available": int(values.shape[0]),
                "relative_best_so_far_mean": mean,
                "relative_best_so_far_sem": sem,
            }
        )
    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], categories=METHODS, ordered=True)
    return summary.sort_values(["method", "time_sec"])


def write_fixed_latex(summary, path):
    lines = []
    for _, row in summary.iterrows():
        time_label = row["time_sec"]
        if time_label != "final":
            time_label = f"{float(time_label):g}"
        lines.append(
            f"{time_label} & {row['method']} & "
            f"{int(row['n_available'])}/{int(row['n_repeats'])} & "
            f"{fmt_mean_sem(row['relative_best_so_far_mean'], row['relative_best_so_far_sem'], 3)} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fixed_wide(summary):
    rows = []
    ordered_times = [t for t in summary["time_sec"].drop_duplicates()]
    ordered_times = sorted([t for t in ordered_times if t != "final"], key=float) + ["final"]
    for time_sec in ordered_times:
        row = {"time_sec": time_sec}
        for method in METHODS:
            match = summary[(summary["time_sec"] == time_sec) & (summary["method"] == method)]
            if match.empty:
                row[f"{method}_mean"] = np.nan
                row[f"{method}_sem"] = np.nan
                row[f"{method}_n"] = 0
                continue
            item = match.iloc[0]
            row[f"{method}_mean"] = item["relative_best_so_far_mean"]
            row[f"{method}_sem"] = item["relative_best_so_far_sem"]
            row[f"{method}_n"] = item["n_available"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_fixed_wide_latex(wide, path):
    lines = []
    for _, row in wide.iterrows():
        time_label = row["time_sec"]
        if time_label != "final":
            time_label = f"{float(time_label):g}"
        cells = [time_label]
        for method in METHODS:
            cells.append(fmt_mean_sem(row[f"{method}_mean"], row[f"{method}_sem"], 3))
        lines.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curve(curve_summary, output_path, targets):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method in METHODS:
        group = curve_summary[curve_summary["method"] == method].sort_values("time_sec")
        ax.plot(
            group["time_sec"],
            group["relative_best_so_far_mean"],
            label=METHOD_LABELS[method],
            color=COLORS[method],
            linewidth=2.0,
        )
        lower = group["relative_best_so_far_mean"] - group["relative_best_so_far_sem"].fillna(0)
        upper = group["relative_best_so_far_mean"] + group["relative_best_so_far_sem"].fillna(0)
        ax.fill_between(
            group["time_sec"].to_numpy(dtype=float),
            lower.to_numpy(dtype=float),
            upper.to_numpy(dtype=float),
            color=COLORS[method],
            alpha=0.13,
            linewidth=0,
        )

    for target in targets:
        ax.axhline(target, color="#333333", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.text(
            0.985,
            target + 0.0015,
            f"{target:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Wall-clock elapsed time (s)")
    ax.set_ylabel("Relative best-so-far validation F1")
    ax.set_title("Wall-Clock Validation Progress on NSL-KDD")
    ax.set_ylim(0.90, 1.005)
    ax.grid(axis="both", linewidth=0.4, alpha=0.3)
    ax.legend(ncol=3, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("NSL-KDD/results_efficiency_grid10/nsl_kdd_ocsvm_search_history.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("NSL-KDD/results_efficiency_grid10"),
    )
    parser.add_argument("--fixed-times", nargs="+", type=float, default=[1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--curve-points", type=int, default=300)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.98, 0.99, 0.995])
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    full = build_full_history(history)
    max_time = float(full["elapsed_sec"].max())
    time_grid = np.linspace(0.0, max_time, args.curve_points)

    snapshot = summarize_fixed_times(full, args.fixed_times)
    snapshot_summary = summarize_fixed_by_method(snapshot)
    snapshot_wide = build_fixed_wide(snapshot_summary)
    curve = build_curve(full, time_grid)
    curve_summary = summarize_curve(curve)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output_dir / "nsl_kdd_ocsvm_fixed_time_relative_f1.csv"
    snapshot_summary_path = args.output_dir / "nsl_kdd_ocsvm_fixed_time_relative_f1_summary.csv"
    snapshot_wide_path = args.output_dir / "nsl_kdd_ocsvm_fixed_time_relative_f1_wide.csv"
    latex_path = args.output_dir / "nsl_kdd_ocsvm_fixed_time_relative_f1_latex_rows.txt"
    wide_latex_path = args.output_dir / "nsl_kdd_ocsvm_fixed_time_relative_f1_wide_latex_rows.txt"
    curve_path = args.output_dir / "nsl_kdd_ocsvm_wallclock_progress_curve.csv"
    curve_summary_path = args.output_dir / "nsl_kdd_ocsvm_wallclock_progress_curve_summary.csv"
    plot_path = args.output_dir / "nsl_kdd_ocsvm_wallclock_progress_curve.png"

    snapshot.to_csv(snapshot_path, index=False)
    snapshot_summary.to_csv(snapshot_summary_path, index=False)
    snapshot_wide.to_csv(snapshot_wide_path, index=False)
    write_fixed_latex(snapshot_summary, latex_path)
    write_fixed_wide_latex(snapshot_wide, wide_latex_path)
    curve.to_csv(curve_path, index=False)
    curve_summary.to_csv(curve_summary_path, index=False)
    plot_curve(curve_summary, plot_path, args.targets)

    print(f"Saved: {snapshot_path}")
    print(f"Saved: {snapshot_summary_path}")
    print(f"Saved: {snapshot_wide_path}")
    print(f"Saved: {latex_path}")
    print(f"Saved: {wide_latex_path}")
    print(f"Saved: {curve_path}")
    print(f"Saved: {curve_summary_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {plot_path.with_suffix('.pdf')}")
    print(snapshot_summary.to_string(index=False))


if __name__ == "__main__":
    main()
