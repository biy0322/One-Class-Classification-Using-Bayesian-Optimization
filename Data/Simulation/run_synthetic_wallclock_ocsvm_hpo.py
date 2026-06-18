import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.ocsvm_hpo import (  # noqa: E402
    METHODS,
    bayes_search,
    grid_search,
    hyperband_search,
    random_search,
    summarize_history,
)


def generate_overlap_data(
    n_samples,
    outlier_fraction,
    separation,
    normal_std,
    anomaly_std,
    seed,
):
    rng = np.random.default_rng(seed)
    n_normal = int((1.0 - outlier_fraction) * n_samples)
    n_abnormal = n_samples - n_normal

    x_normal = rng.normal(loc=0.0, scale=normal_std, size=(n_normal, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_abnormal)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * separation
    x_abnormal = centers + rng.normal(loc=0.0, scale=anomaly_std, size=(n_abnormal, 2))

    x = np.vstack([x_normal, x_abnormal])
    y = np.r_[np.zeros(n_normal, dtype=int), np.ones(n_abnormal, dtype=int)]
    return train_test_split(x, y, stratify=y, test_size=0.2, random_state=seed)


def split_and_transform(x_train_raw, y_train, x_test_raw, y_test, seed, val_size):
    idx = np.arange(len(y_train))
    fit_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        stratify=y_train,
        random_state=seed + 1000,
    )
    normal_fit_idx = fit_idx[y_train[fit_idx] == 0]

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor.fit(x_train_raw[normal_fit_idx])

    x_fit = preprocessor.transform(x_train_raw[normal_fit_idx]).astype(np.float32)
    x_val = preprocessor.transform(x_train_raw[val_idx]).astype(np.float32)
    x_test = preprocessor.transform(x_test_raw).astype(np.float32)

    return x_fit, x_val, y_train[val_idx], x_test, y_test


def predict_ocsvm(model, x):
    raw = model.predict(x)
    return np.where(raw == -1, 1, 0)


def metric_bundle(y_true, y_pred):
    return {
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F-1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC": roc_auc_score(y_true, y_pred),
    }


def fit_and_test(x_fit, x_test, y_test, gamma, nu):
    model = OneClassSVM(kernel="rbf", gamma=float(gamma), nu=float(nu))
    model.fit(x_fit)
    pred = predict_ocsvm(model, x_test)
    return metric_bundle(y_test, pred)


def summarize_metrics(results, selected_methods):
    return (
        results.groupby("method")[["Recall", "F-1", "AUC"]]
        .agg(["mean", "sem"])
        .reindex(selected_methods)
    )


def run_experiment(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = list(dict.fromkeys(args.methods))
    effective_n_trials = args.grid_size * args.grid_size if args.equal_budget_from_grid else args.n_trials

    meta = {
        "dataset": "moderate_overlap_synthetic",
        "n_samples": args.n_samples,
        "outlier_fraction": args.outlier_fraction,
        "normal_to_anomaly_ratio": f"{int((1.0 - args.outlier_fraction) * 100)}:{int(args.outlier_fraction * 100)}",
        "separation": args.separation,
        "normal_std": args.normal_std,
        "anomaly_std": args.anomaly_std,
        "n_repeats": args.n_repeats,
        "n_trials": effective_n_trials,
        "grid_size": args.grid_size,
        "grid_budget": args.grid_size * args.grid_size,
        "equal_budget_from_grid": bool(args.equal_budget_from_grid),
        "test_size": 0.2,
        "val_size": args.val_size,
        "validation_metric": args.validation_metric,
        "validation_beta": args.validation_beta,
        "validation_average": args.validation_average,
        "methods": ",".join(selected_methods),
    }
    print("metadata", meta, flush=True)

    rows = []
    history_rows = []
    time_rows = []
    split_rows = []

    for rep in range(args.n_repeats):
        seed = args.seed + rep
        x_train_raw, x_test_raw, y_train, y_test = generate_overlap_data(
            args.n_samples,
            args.outlier_fraction,
            args.separation,
            args.normal_std,
            args.anomaly_std,
            seed,
        )
        x_fit, x_val, y_val, x_test, y_test = split_and_transform(
            x_train_raw,
            y_train,
            x_test_raw,
            y_test,
            seed,
            args.val_size,
        )
        rng = np.random.default_rng(seed)
        split_rows.append(
            {
                "repeat": rep,
                "train": int(len(y_train)),
                "train_abnormal": int(y_train.sum()),
                "fit_normals": int(len(x_fit)),
                "val": int(len(y_val)),
                "val_abnormal": int(y_val.sum()),
                "test": int(len(y_test)),
                "test_abnormal": int(y_test.sum()),
            }
        )
        print(
            f"repeat {rep + 1}/{args.n_repeats}: fit_normals={len(x_fit)}, "
            f"val={len(y_val)}, val_abnormal={int(y_val.sum())}, "
            f"test={len(y_test)}, test_abnormal={int(y_test.sum())}",
            flush=True,
        )

        searches = {
            "GC": lambda: grid_search(
                x_fit,
                x_val,
                y_val,
                args.grid_size,
                validation_metric=args.validation_metric,
                validation_beta=args.validation_beta,
                validation_average=args.validation_average,
            ),
            "RC": lambda: random_search(
                x_fit,
                x_val,
                y_val,
                rng,
                effective_n_trials,
                validation_metric=args.validation_metric,
                validation_beta=args.validation_beta,
                validation_average=args.validation_average,
            ),
            "BO": lambda: bayes_search(
                x_fit,
                x_val,
                y_val,
                rng,
                effective_n_trials,
                validation_metric=args.validation_metric,
                validation_beta=args.validation_beta,
                validation_average=args.validation_average,
            ),
            "HB": lambda: hyperband_search(
                x_fit,
                x_val,
                y_val,
                seed,
                effective_n_trials,
                "random",
                validation_metric=args.validation_metric,
                validation_beta=args.validation_beta,
                validation_average=args.validation_average,
            ),
            "BOHB": lambda: hyperband_search(
                x_fit,
                x_val,
                y_val,
                seed,
                effective_n_trials,
                "tpe",
                validation_metric=args.validation_metric,
                validation_beta=args.validation_beta,
                validation_average=args.validation_average,
            ),
        }

        for method in selected_methods:
            start = time.time()
            params, history = searches[method]()
            metrics = fit_and_test(x_fit, x_test, y_test, **params)
            elapsed = time.time() - start
            time_summary = summarize_history(history)
            row = {
                "repeat": rep,
                "method": method,
                "validation_metric": args.validation_metric,
                "validation_beta": args.validation_beta,
                "validation_average": args.validation_average,
                "gamma": params["gamma"],
                "nu": params["nu"],
                "elapsed_sec": elapsed,
                "best_validation_score": time_summary["best_validation_score"],
                "configs_to_best": time_summary["configs_to_best"],
                "observed_evals_to_best": time_summary["observed_evals_to_best"],
                "time_to_best_sec": time_summary["time_to_best_sec"],
                "total_observed_evals": time_summary["total_observed_evals"],
                "total_time_sec": time_summary["total_time_sec"],
                **metrics,
            }
            rows.append(row)
            time_rows.append(
                {
                    "repeat": rep,
                    "method": method,
                    "validation_metric": args.validation_metric,
                    "validation_beta": args.validation_beta,
                    "validation_average": args.validation_average,
                    **time_summary,
                }
            )
            for h in history:
                history_rows.append(
                    {
                        "repeat": rep,
                        "method": method,
                        "validation_metric": args.validation_metric,
                        "validation_beta": args.validation_beta,
                        "validation_average": args.validation_average,
                        **h,
                    }
                )
            print(
                f"  {method}: gamma={params['gamma']:.6g}, nu={params['nu']:.4f}, "
                f"Recall={metrics['Recall']:.4f}, F-1={metrics['F-1']:.4f}, "
                f"AUC={metrics['AUC']:.4f}, total={elapsed:.2f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    time_to_best = pd.DataFrame(time_rows)
    history = pd.DataFrame(history_rows)
    splits = pd.DataFrame(split_rows)
    summary = summarize_metrics(results, selected_methods)
    time_summary = (
        time_to_best.groupby("method")[
            [
                "best_validation_score",
                "configs_to_best",
                "observed_evals_to_best",
                "time_to_best_sec",
                "total_configurations",
                "full_validation_evals",
                "total_observed_evals",
                "total_time_sec",
                "total_evaluation_time_sec",
                "optimization_overhead_sec",
                "optimization_overhead_pct",
            ]
        ]
        .agg(["mean", "sem"])
        .reindex(selected_methods)
    )

    results.to_csv(args.output_dir / "synthetic_ocsvm_repeated_results.csv", index=False)
    history.to_csv(args.output_dir / "synthetic_ocsvm_search_history.csv", index=False)
    time_to_best.to_csv(args.output_dir / "synthetic_ocsvm_time_to_best.csv", index=False)
    splits.to_csv(args.output_dir / "synthetic_ocsvm_splits.csv", index=False)
    summary.to_csv(args.output_dir / "synthetic_ocsvm_summary.csv")
    time_summary.to_csv(args.output_dir / "synthetic_ocsvm_time_summary.csv")
    pd.DataFrame([meta]).to_csv(args.output_dir / "synthetic_ocsvm_metadata.csv", index=False)

    print("\nsummary")
    print(summary)
    print("\ntime summary")
    print(time_summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("Simulation/results_synthetic_wallclock"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--outlier-fraction", type=float, default=0.05)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--normal-std", type=float, default=0.5)
    parser.add_argument("--anomaly-std", type=float, default=0.6)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--equal-budget-from-grid", action="store_true")
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--validation-metric", choices=["f1", "f_beta"], default="f_beta")
    parser.add_argument("--validation-beta", type=float, default=1.0)
    parser.add_argument("--validation-average", choices=["binary", "macro"], default="binary")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
