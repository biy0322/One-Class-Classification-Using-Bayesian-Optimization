#!/usr/bin/env python
# coding: utf-8

"""Summarize overlap simulation metrics and search time across scenarios."""

from pathlib import Path
import argparse
import re

import pandas as pd


def parse_separation(name):
    match = re.search(r"sep_([0-9]+p[0-9]+)", name)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def parse_outlier_fraction(name):
    match = re.search(r"ratio_([0-9]+p[0-9]+)", name)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def format_ratio(outlier_fraction):
    if outlier_fraction is None:
        return None
    normal = int(round((1.0 - outlier_fraction) * 100))
    anomaly = int(round(outlier_fraction * 100))
    return f"{normal}:{anomaly}"


def summarize_timing(timing):
    rows = []
    for method, group in timing.groupby("method"):
        rows.append({
            "method": method,
            "time_mean_sec": group["wall_time_sec"].mean(),
            "time_std_sec": group["wall_time_sec"].std(ddof=1),
            "time_sem_sec": group["wall_time_sec"].sem(ddof=1),
            "time_total_sec": group["wall_time_sec"].sum(),
            "n_timing_rows": len(group),
            "budget_evaluations": group["budget_evaluations"].iloc[0],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "Review_Overlap")
    args = parser.parse_args()

    all_rows = []
    for scenario_dir in sorted(args.root.glob("overlap_*")):
        if not scenario_dir.is_dir():
            continue
        result_dir = scenario_dir / "Result"
        summary_path = result_dir / "summary.csv"
        timing_path = result_dir / "timing.csv"

        if not summary_path.exists():
            print(f"Skip without summary.csv: {scenario_dir.name}")
            continue

        summary = pd.read_csv(summary_path)
        summary["scenario"] = scenario_dir.name
        summary["outlier_fraction"] = parse_outlier_fraction(scenario_dir.name)
        summary["normal_to_anomaly_ratio"] = summary["outlier_fraction"].apply(format_ratio)
        summary["separation"] = parse_separation(scenario_dir.name)

        if timing_path.exists():
            timing_summary = summarize_timing(pd.read_csv(timing_path))
            summary = summary.merge(timing_summary, on="method", how="left")
        else:
            summary["time_mean_sec"] = pd.NA
            summary["time_std_sec"] = pd.NA
            summary["time_sem_sec"] = pd.NA
            summary["time_total_sec"] = pd.NA
            summary["n_timing_rows"] = pd.NA
            summary["budget_evaluations"] = pd.NA

        all_rows.append(summary)

    if not all_rows:
        raise FileNotFoundError(f"No scenario summaries found under {args.root}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[
        [
            "scenario",
            "outlier_fraction",
            "normal_to_anomaly_ratio",
            "separation",
            "metric",
            "method",
            "mean",
            "std",
            "sem",
            "min",
            "max",
            "time_mean_sec",
            "time_sem_sec",
            "time_total_sec",
            "budget_evaluations",
        ]
    ]
    combined = combined.sort_values(["separation", "metric", "method"], ascending=[False, True, True])

    output_path = args.root / "overlap_summary_with_time.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(combined)


if __name__ == "__main__":
    main()
