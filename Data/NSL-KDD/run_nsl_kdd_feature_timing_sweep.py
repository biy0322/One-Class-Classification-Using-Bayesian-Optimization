import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run_command(cmd):
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def sem(series):
    return series.std(ddof=1) / (len(series) ** 0.5) if len(series) > 1 else 0.0


def summarize(base_dir, feature_labels):
    rows = []
    for label in feature_labels:
        run_dir = base_dir / f"features_{label}"
        time_path = run_dir / "nsl_kdd_ocsvm_time_to_best.csv"
        result_path = run_dir / "nsl_kdd_ocsvm_repeated_results.csv"
        if not time_path.exists() or not result_path.exists():
            continue
        time_df = pd.read_csv(time_path)
        result_df = pd.read_csv(result_path)
        merged = time_df.merge(
            result_df[
                [
                    "repeat",
                    "method",
                    "F-1",
                    "AUC",
                    "Recall",
                    "transformed_features",
                    "original_transformed_features",
                ]
            ],
            on=["repeat", "method", "transformed_features", "original_transformed_features"],
            how="left",
        )
        for method, group in merged.groupby("method"):
            rows.append(
                {
                    "feature_setting": label,
                    "method": method,
                    "transformed_features_mean": group["transformed_features"].mean(),
                    "original_transformed_features_mean": group["original_transformed_features"].mean(),
                    "total_time_sec_mean": group["total_time_sec"].mean(),
                    "total_time_sec_sem": sem(group["total_time_sec"]),
                    "total_evaluation_time_sec_mean": group["total_evaluation_time_sec"].mean(),
                    "total_evaluation_time_sec_sem": sem(group["total_evaluation_time_sec"]),
                    "total_fit_time_sec_mean": group["total_fit_time_sec"].mean(),
                    "total_fit_time_sec_sem": sem(group["total_fit_time_sec"]),
                    "total_predict_time_sec_mean": group["total_predict_time_sec"].mean(),
                    "total_predict_time_sec_sem": sem(group["total_predict_time_sec"]),
                    "total_score_time_sec_mean": group["total_score_time_sec"].mean(),
                    "total_score_time_sec_sem": sem(group["total_score_time_sec"]),
                    "total_surrogate_fit_time_sec_mean": group["total_surrogate_fit_time_sec"].mean(),
                    "total_surrogate_fit_time_sec_sem": sem(group["total_surrogate_fit_time_sec"]),
                    "total_acquisition_time_sec_mean": group["total_acquisition_time_sec"].mean(),
                    "total_acquisition_time_sec_sem": sem(group["total_acquisition_time_sec"]),
                    "optimization_overhead_sec_mean": group["optimization_overhead_sec"].mean(),
                    "optimization_overhead_sec_sem": sem(group["optimization_overhead_sec"]),
                    "time_to_best_sec_mean": group["time_to_best_sec"].mean(),
                    "time_to_best_sec_sem": sem(group["time_to_best_sec"]),
                    "configs_to_best_mean": group["configs_to_best"].mean(),
                    "configs_to_best_sem": sem(group["configs_to_best"]),
                    "f1_mean": group["F-1"].mean(),
                    "f1_sem": sem(group["F-1"]),
                    "auc_mean": group["AUC"].mean(),
                    "auc_sem": sem(group["AUC"]),
                    "recall_mean": group["Recall"].mean(),
                    "recall_sem": sem(group["Recall"]),
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(["transformed_features_mean", "method"])
    summary.to_csv(base_dir / "feature_timing_summary.csv", index=False)
    return summary


def plot_summary(summary, base_dir):
    if summary.empty:
        return
    colors = {"GC": "#2f7d32", "RC": "#f2b134", "BO": "#7a1f87"}
    for metric, ylabel, filename in [
        ("total_time_sec_mean", "Total search time [s]", "feature_count_total_time.png"),
        ("time_to_best_sec_mean", "Time to best validation F1 [s]", "feature_count_time_to_best.png"),
        ("total_fit_time_sec_mean", "Accumulated OC-SVM fit time [s]", "feature_count_fit_time.png"),
    ]:
        plt.figure(figsize=(6.2, 4.0))
        for method, group in summary.groupby("method"):
            group = group.sort_values("transformed_features_mean")
            plt.plot(
                group["transformed_features_mean"],
                group[metric],
                marker="o",
                linewidth=2,
                label=method,
                color=colors.get(method),
            )
        plt.xlabel("Number of preprocessed features")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(base_dir / filename, dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("NSL-KDD/results_feature_count_timing"))
    parser.add_argument("--feature-counts", nargs="+", default=["10", "20", "40", "80", "all"])
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--target-train-size", type=int, default=8000)
    parser.add_argument("--target-test-size", type=int, default=10000)
    parser.add_argument("--target-attack-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_nsl_kdd_ocsvm_hpo.py")
    for label in args.feature_counts:
        run_dir = args.output_dir / f"features_{label}"
        done_path = run_dir / "nsl_kdd_ocsvm_time_to_best.csv"
        if args.skip_existing and done_path.exists():
            print("Skipping existing:", run_dir, flush=True)
            continue
        cmd = [
            sys.executable,
            str(runner),
            "--output-dir",
            str(run_dir),
            "--n-repeats",
            str(args.n_repeats),
            "--grid-size",
            str(args.grid_size),
            "--equal-budget-from-grid",
            "--max-fit-normals",
            "10000",
            "--max-val-samples",
            "10000",
            "--val-size",
            "0.25",
            "--seed",
            str(args.seed),
            "--target-attack-rate",
            str(args.target_attack_rate),
            "--target-train-size",
            str(args.target_train_size),
            "--target-test-size",
            str(args.target_test_size),
            "--validation-metric",
            "f_beta",
            "--validation-beta",
            "1.0",
            "--validation-average",
            "binary",
            "--methods",
            "GC",
            "RC",
            "BO",
        ]
        if label != "all":
            cmd.extend(["--transformed-feature-count", label])
        run_command(cmd)

    summary = summarize(args.output_dir, args.feature_counts)
    plot_summary(summary, args.output_dir)
    if not summary.empty:
        print("\nfeature timing summary")
        print(
            summary[
                [
                    "feature_setting",
                    "method",
                    "transformed_features_mean",
                    "total_time_sec_mean",
                    "total_fit_time_sec_mean",
                    "time_to_best_sec_mean",
                    "f1_mean",
                    "auc_mean",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
