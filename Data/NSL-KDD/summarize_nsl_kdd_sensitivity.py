import argparse
from pathlib import Path

import pandas as pd


def flatten_multiindex(frame):
    flat = frame.copy()
    flat.columns = [f"{metric}_{stat}" for metric, stat in flat.columns]
    flat = flat.reset_index().rename(columns={"method": "method", "index": "method"})
    return flat


def read_performance_summary(path):
    frame = pd.read_csv(path, header=[0, 1], index_col=0)
    frame.index.name = "method"
    return flatten_multiindex(frame)


def read_time_summary(path):
    frame = pd.read_csv(path, header=[0, 1], index_col=0)
    frame.index.name = "method"
    return flatten_multiindex(frame)


def read_metadata(path):
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def collect(
    root,
    include_smoke=False,
    include_legacy_kernel_sweep=False,
    include_time=False,
    experiment_group_filter=None,
):
    rows = []
    for summary_path in sorted(root.glob("**/nsl_kdd_ocsvm_summary.csv")):
        run_dir = summary_path.parent
        rel = run_dir.relative_to(root)
        parts = rel.parts
        if not include_smoke and any("smoke" in part.lower() for part in parts):
            continue
        if not include_legacy_kernel_sweep and parts and parts[0] == "kernel_sweep":
            continue
        meta = read_metadata(run_dir / "nsl_kdd_ocsvm_metadata.csv")
        perf = read_performance_summary(summary_path)
        time_path = run_dir / "nsl_kdd_ocsvm_time_summary.csv"
        if include_time and time_path.exists():
            time_summary = read_time_summary(time_path)
            merged = perf.merge(time_summary, on="method", suffixes=("", "_time"))
        else:
            merged = perf

        experiment_group = parts[0] if parts else ""
        if experiment_group_filter and experiment_group not in experiment_group_filter:
            continue
        experiment_name = parts[1] if len(parts) > 1 else rel.as_posix()
        for _, row in merged.iterrows():
            out = {
                "experiment_group": experiment_group,
                "experiment_name": experiment_name,
                "run_dir": rel.as_posix(),
                "method": row["method"],
                "validation_metric": meta.get("validation_metric", ""),
                "validation_beta": meta.get("validation_beta", ""),
                "validation_average": meta.get("validation_average", "macro") or "macro",
                "bo_kernel": meta.get("bo_kernel", ""),
                "n_repeats": meta.get("n_repeats", ""),
                "n_trials": meta.get("n_trials", ""),
                "grid_size": meta.get("grid_size", ""),
            }
            out.update({k: v for k, v in row.items() if k != "method"})
            rows.append(out)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("NSL-KDD/results_sensitivity_binary_f1"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--include-legacy-kernel-sweep", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--experiment-group", nargs="+", default=None)
    args = parser.parse_args()

    summary = collect(
        args.root,
        include_smoke=args.include_smoke,
        include_legacy_kernel_sweep=args.include_legacy_kernel_sweep,
        include_time=args.include_time,
        experiment_group_filter=args.experiment_group,
    )
    output = args.output or (args.root / "nsl_kdd_sensitivity_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved: {output}")
    if not summary.empty:
        display_cols = [
            "experiment_group",
            "experiment_name",
            "method",
            "validation_beta",
            "validation_average",
            "bo_kernel",
            "F-1_mean",
            "F-1_sem",
            "AUC_mean",
            "AUC_sem",
        ]
        display_cols = [col for col in display_cols if col in summary.columns]
        print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
