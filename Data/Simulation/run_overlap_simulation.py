#!/usr/bin/env python
# coding: utf-8

"""Run review-oriented synthetic experiments with ambiguous OCC boundaries.

Example:
    python run_overlap_simulation.py --separations 2.0 1.25 0.75 0.5 --outlier-fraction 0.05

For a quick smoke run:
    python run_overlap_simulation.py --n-repeats 2 --n-splits 2 --separations 1.0
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from synthetic_dataset import Dataset
from Train import Train
from Evaluation import predict


def scenario_name(outlier_fraction, separation, normal_std, anomaly_std):
    name = (
        f"overlap_ratio_{outlier_fraction:.2f}_"
        f"sep_{separation:.2f}_"
        f"nstd_{normal_std:.2f}_"
        f"astd_{anomaly_std:.2f}"
    )
    return name.replace(".", "p")


def summarize_metric_frames(metric_frames):
    rows = []
    for metric_name, frame in metric_frames.items():
        for method in frame.columns:
            values = frame[method]
            rows.append({
                "metric": metric_name,
                "method": method,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "sem": values.sem(ddof=1),
                "min": values.min(),
                "max": values.max(),
            })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument(
        "--outlier-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of outlier fractions. Overrides --outlier-fraction when provided.",
    )
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--separations", nargs="+", type=float, default=[2.0, 1.25, 0.75, 0.5])
    parser.add_argument("--normal-std", type=float, default=0.5)
    parser.add_argument("--anomaly-std", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--kappa", type=float, default=15.0)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "Review_Overlap")
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    outlier_fractions = args.outlier_fractions or [args.outlier_fraction]

    for outlier_fraction in outlier_fractions:
        for separation in args.separations:
            name = scenario_name(outlier_fraction, separation, args.normal_std, args.anomaly_std)
            scenario_dir = args.output_root / name
            data_dir = scenario_dir / "Data"
            model_dir = scenario_dir / "Model"
            result_dir = scenario_dir / "Result"
            data_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)

            print(f"=== Scenario: {name} ===")
            dataset = Dataset()
            X_train, y_train, X_test, y_test = dataset.dataset(
                n_samples=args.n_samples,
                outlier_fraction=outlier_fraction,
                n_repeats=args.n_repeats,
                generator="overlap",
                random_state=args.seed,
                separation=separation,
                normal_std=args.normal_std,
                anomaly_std=args.anomaly_std,
            )

            np.save(data_dir / "X_train_dataset.npy", np.array(X_train))
            np.save(data_dir / "y_train_dataset.npy", np.array(y_train))
            np.save(data_dir / "X_test_dataset.npy", np.array(X_test))
            np.save(data_dir / "y_test_dataset.npy", np.array(y_test))

            metadata = pd.DataFrame([{
                "n_samples": args.n_samples,
                "outlier_fraction": outlier_fraction,
                "normal_to_anomaly_ratio": f"{int((1 - outlier_fraction) * 100)}:{int(outlier_fraction * 100)}",
                "n_repeats": args.n_repeats,
                "n_splits": args.n_splits,
                "separation": separation,
                "normal_std": args.normal_std,
                "anomaly_std": args.anomaly_std,
                "seed": args.seed,
                "tuning_scoring": "f_beta",
                "tuning_beta": args.beta,
                "kappa": args.kappa,
            }])
            metadata.to_csv(result_dir / "metadata.csv", index=False)

            if args.generate_only:
                print("Generated data only. Skip training/evaluation.")
                continue

            save_prefix = str(model_dir / "model")
            timing_path = result_dir / "timing.csv"
            history_path = result_dir / "search_history.csv"

            trainer = Train(
                X_train,
                X_test,
                y_train,
                y_test,
                save_path=save_prefix,
                scoring="f_beta",
                beta=args.beta,
                kappa=args.kappa,
            )
            trainer.train(
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                timing_path=timing_path,
                history_path=history_path,
            )

            recall_df, f1_df, roc_auc_df = predict(
                n_splits=args.n_splits,
                x_test_dataset=X_test,
                y_test_dataset=y_test,
                save_path=save_prefix,
                n_repeats=args.n_repeats,
            )

            recall_df.to_csv(result_dir / "recall.csv", index=False)
            f1_df.to_csv(result_dir / "f1.csv", index=False)
            roc_auc_df.to_csv(result_dir / "roc_auc.csv", index=False)

            summary = summarize_metric_frames({
                "recall": recall_df,
                "f1": f1_df,
                "roc_auc": roc_auc_df,
            })
            summary.to_csv(result_dir / "summary.csv", index=False)
            print(summary)


if __name__ == "__main__":
    main()
