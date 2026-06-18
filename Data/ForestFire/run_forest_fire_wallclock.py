import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from Dataset import Dataset
from Hopt import Bayesian_hopt, BOHB_hopt, GridSearch, Hyperband_hopt, RandomSearch


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]


def majority_vote(models, x):
    preds = [model.predict(x) for model in models]
    return (np.mean([pred.astype(int) for pred in preds], axis=0) > 0.5).astype(int)


def evaluate(y_true, y_pred):
    return {
        "Recall": recall_score(y_true, y_pred),
        "F-1": f1_score(y_true, y_pred, average="macro"),
        "AUC": roc_auc_score(y_true, y_pred),
    }


def load_models(output_dir, prefix, method, n_splits):
    suffix = {
        "GC": "grid",
        "RC": "random",
        "BO": "bayes",
        "HB": "hyperband",
        "BOHB": "bohb",
    }[method]
    models = []
    for fold in range(n_splits):
        with open(output_dir / f"{prefix}_{suffix}_cv_{fold}", "rb") as f:
            models.append(pickle.load(f))
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, default=Path("data_manipulated.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_wallclock_original"))
    parser.add_argument("--prefix", default="forest_fire")
    parser.add_argument("--n-splits", type=int, default=9)
    parser.add_argument("--scoring", default="f_beta", choices=["recall", "f_beta"])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=15.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(load_path=args.data_file)
    x_folds, x_valid_folds, y_folds, y_valid_folds = dataset.train_valid_set(cv=args.n_splits)

    prefix_path = args.output_dir / args.prefix
    rows = []

    for fold in range(args.n_splits):
        x_train = x_folds[fold]
        x_valid = x_valid_folds[fold]
        y_train = y_folds[fold]
        y_valid = y_valid_folds[fold]

        print(f"fold {fold + 1}/{args.n_splits}", flush=True)

        default_start = time.perf_counter()
        default_model = OCSVM(kernel="rbf")
        default_model.fit(x_train)
        with open(args.output_dir / f"{args.prefix}_model_cv_{fold}", "wb") as f:
            pickle.dump(default_model, f)
        default_elapsed = time.perf_counter() - default_start

        searches = {
            "GC": GridSearch(
                x_train,
                x_valid,
                y_train,
                y_valid,
                str(prefix_path),
                scoring=args.scoring,
                beta=args.beta,
            ),
            "RC": RandomSearch(
                x_train,
                x_valid,
                y_train,
                y_valid,
                str(prefix_path),
                scoring=args.scoring,
                beta=args.beta,
            ),
            "BO": Bayesian_hopt(
                x_train,
                x_valid,
                y_train,
                y_valid,
                str(prefix_path),
                scoring=args.scoring,
                utility="ucb",
                kappa=args.kappa,
                beta=args.beta,
            ),
            "HB": Hyperband_hopt(
                x_train,
                x_valid,
                y_train,
                y_valid,
                str(prefix_path),
                scoring=args.scoring,
                beta=args.beta,
            ),
            "BOHB": BOHB_hopt(
                x_train,
                x_valid,
                y_train,
                y_valid,
                str(prefix_path),
                scoring=args.scoring,
                beta=args.beta,
            ),
        }

        rows.append(
            {
                "fold": fold,
                "method": "Default",
                "elapsed_sec": default_elapsed,
                "n_train_normals": len(x_train),
                "n_valid": len(y_valid),
                "n_valid_abnormal": int(np.sum(y_valid)),
            }
        )

        for method, search in searches.items():
            start = time.perf_counter()
            search.fit(fold=fold)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "elapsed_sec": elapsed,
                    "n_train_normals": len(x_train),
                    "n_valid": len(y_valid),
                    "n_valid_abnormal": int(np.sum(y_valid)),
                }
            )
            print(f"  {method}: {elapsed:.3f}s", flush=True)

    timing = pd.DataFrame(rows)
    timing.to_csv(args.output_dir / f"{args.prefix}_wallclock_by_fold.csv", index=False)

    summary = (
        timing.groupby("method")["elapsed_sec"]
        .agg(["mean", "sem", "sum"])
        .reindex(["Default", *METHODS])
        .reset_index()
    )
    summary.to_csv(args.output_dir / f"{args.prefix}_wallclock_summary.csv", index=False)

    performance_rows = []
    for method in METHODS:
        models = load_models(args.output_dir, args.prefix, method, args.n_splits)
        pred = majority_vote(models, dataset.X)
        performance_rows.append({"method": method, **evaluate(dataset.y, pred)})
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(args.output_dir / f"{args.prefix}_performance_summary.csv", index=False)

    print("\nWall-clock summary")
    print(summary.to_string(index=False))
    print("\nPerformance summary")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
