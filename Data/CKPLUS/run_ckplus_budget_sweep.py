#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
import time
from itertools import product
from pathlib import Path

OCC_SITE_PACKAGES = Path.home() / "anaconda3" / "envs" / "OCC" / "Lib" / "site-packages"
if OCC_SITE_PACKAGES.exists() and str(OCC_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(OCC_SITE_PACKAGES))

import numpy as np
import torch
from sklearn.metrics import fbeta_score

from network.evaluation import eval as svdd_eval
from network.hopt import Bayesian, BOHB_hopt, Hyperband_hopt
from network.Train import TrainerDeepSVDD

from run_ckplus_wallclock import (
    METHODS,
    evaluate_method,
    load_class_images,
    make_args,
    make_ckplus_protocol_dataset,
    make_loaders,
    read_seeds,
    resolve_device,
    set_seed,
    split_dataset,
    sync_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CK+ budget sweep for Deep SVDD HPO performance and wall-clock time."
    )
    parser.add_argument("--data-root", default="data/CK")
    parser.add_argument("--output-dir", default="results_budget_sweep")
    parser.add_argument("--seed-file", default="seed_list.txt")
    parser.add_argument("--max-seeds", type=int, default=1)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Number of hyperparameter candidates/trials per method.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-epoch", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=5e-7)
    parser.add_argument("--bo-kappa", type=float, default=15.0)
    parser.add_argument("--max-resource", type=int, default=30)
    parser.add_argument("--min-resource", type=int, default=1)
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--abnormal-classes",
        nargs="+",
        default=["anger", "disgust", "fear", "sadness"],
    )
    parser.add_argument("--normal-count", type=int, default=1140)
    parser.add_argument("--abnormal-count", type=int, default=60)
    parser.add_argument("--augmentation-iterations", type=int, default=6)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def model_path(output_dir: Path, method: str, simulation: str) -> Path:
    return output_dir / "Model" / method / "best_model" / simulation / f"{simulation}_result.pt"


def ensure_model_dir(output_dir: Path, method: str, simulation: str) -> Path:
    path = output_dir / "Model" / method
    (path / "best_model" / simulation).mkdir(parents=True, exist_ok=True)
    return path


def grid_candidates(budget: int) -> list[tuple[float, float]]:
    levels = int(np.ceil(np.sqrt(budget)))
    lrs = np.linspace(0.0001, 0.01, levels)
    nus = np.linspace(0.001, 0.9999, levels)
    candidates = list(product(lrs, nus))
    if len(candidates) == budget:
        selected = candidates
    else:
        selected_idx = np.linspace(0, len(candidates) - 1, budget, dtype=int)
        selected = [candidates[i] for i in selected_idx]
    return [(float(lr), float(nu)) for lr, nu in selected]


def random_candidates(budget: int, seed: int = 1500) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [
        (rng.uniform(0.0001, 0.01), rng.uniform(0.001, 0.9999))
        for _ in range(budget)
    ]


def fit_candidate_search(
    method: str,
    candidates: list[tuple[float, float]],
    hpo_args,
    train_loader,
    valid_loader,
    output_dir: Path,
    device: torch.device,
    simulation: str,
) -> float:
    method_dir = ensure_model_dir(output_dir, method, simulation)
    best_score = None
    best_model = None

    for lr, nu in candidates:
        trainer = TrainerDeepSVDD(
            hpo_args,
            train_loader,
            device,
            "soft-boundary",
            lr=lr,
            R=0,
            nu=nu,
            warm_up_n_epochs=5,
        )
        net, c, R = trainer.train(hpo_args.early_stopping_epoch)
        true_v, score_v = svdd_eval(net, c, R, "soft-boundary", valid_loader, device)
        pred_v = [0 if score <= 0 else 1 for score in score_v]
        score = fbeta_score(true_v, pred_v, beta=1, zero_division=0)

        if best_score is None or score > best_score:
            best_score = float(score)
            best_model = {
                "net": copy.deepcopy(net.state_dict()),
                "c": c.clone(),
                "R": R.clone(),
                "lr": lr,
                "nu": nu,
                "threshold_1": trainer.threshold_1,
                "threshold_2": trainer.threshold_2,
            }

    if best_model is None:
        raise RuntimeError(f"No model was fitted for {method}.")

    torch.save(best_model, model_path(output_dir, method, simulation))
    return float(best_score)


def fit_existing_search(
    method: str,
    budget: int,
    hpo_args,
    train_loader,
    valid_loader,
    output_dir: Path,
    device: torch.device,
    simulation: str,
    cli: argparse.Namespace,
) -> None:
    method_dir = ensure_model_dir(output_dir, method, simulation)
    path = str(method_dir) + os.sep

    if method == "Bayes":
        n_iter = max(0, budget - 3)
        search = Bayesian(
            hpo_args,
            train_loader,
            valid_loader,
            path=path,
            device=device,
            name=simulation,
            early_stopping_epochs=hpo_args.early_stopping_epoch,
            objective="soft-boundary",
            n_iter=n_iter,
            f_beta_param=1,
            kappa=cli.bo_kappa,
        )
        search.fit(simulation)
    elif method == "Hyperband":
        search = Hyperband_hopt(
            hpo_args,
            train_loader,
            valid_loader,
            path=path,
            device=device,
            objective="soft-boundary",
            f_beta_param=1,
            n_trials=budget,
            max_resource=cli.max_resource,
            min_resource=cli.min_resource,
            reduction_factor=cli.reduction_factor,
        )
        search.fit(simulation)
    elif method == "BOHB":
        search = BOHB_hopt(
            hpo_args,
            train_loader,
            valid_loader,
            path=path,
            device=device,
            objective="soft-boundary",
            f_beta_param=1,
            n_trials=budget,
            max_resource=cli.max_resource,
            min_resource=cli.min_resource,
            reduction_factor=cli.reduction_factor,
        )
        search.fit(simulation)
    else:
        raise ValueError(f"Unsupported existing search method: {method}")


def existing_keys(csv_path: Path) -> set[tuple[int, int, str]]:
    if not csv_path.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    return set(zip(df["seed"].astype(int), df["budget"].astype(int), df["method"].astype(str)))


def append_budget_row(csv_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "seed",
        "budget",
        "method",
        "elapsed_sec",
        "lr",
        "nu",
        "Recall",
        "F1",
        "AUC",
        "device",
        "n_train",
        "n_train_normal",
        "n_train_abnormal",
        "n_val",
        "n_val_normal",
        "n_val_abnormal",
        "n_test",
        "n_test_normal",
        "n_test_abnormal",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize(csv_path: Path, summary_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    grouped = df.groupby(["budget", "method"]).agg(
        elapsed_sec_mean=("elapsed_sec", "mean"),
        elapsed_sec_sem=("elapsed_sec", "sem"),
        Recall_mean=("Recall", "mean"),
        Recall_sem=("Recall", "sem"),
        F1_mean=("F1", "mean"),
        F1_sem=("F1", "sem"),
        AUC_mean=("AUC", "mean"),
        AUC_sem=("AUC", "sem"),
    )
    grouped.to_csv(summary_path)
    print("\nsummary")
    print(grouped)


def main() -> None:
    cli = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_root = (base_dir / cli.data_root).resolve()
    output_dir = (base_dir / cli.output_dir).resolve()
    seed_file = (base_dir / cli.seed_file).resolve()
    csv_path = output_dir / "ckplus_budget_sweep_results.csv"
    summary_path = output_dir / "ckplus_budget_sweep_summary.csv"
    device = resolve_device(cli.device)
    hpo_args = make_args(cli)

    class_images, class_counts = load_class_images(data_root)
    seeds = read_seeds(seed_file, cli.max_seeds)

    print(f"data_root={data_root}")
    print(f"class_counts={class_counts}")
    print(f"abnormal_classes={cli.abnormal_classes}")
    print(f"sampling_protocol=normal {cli.normal_count}, abnormal {cli.abnormal_count}")
    print(f"budgets={cli.budgets}")
    print(f"methods={cli.methods}")
    print(f"device={device}")
    done = existing_keys(csv_path) if cli.skip_existing else set()

    for seed in seeds:
        set_seed(seed)
        x, y, sample_meta = make_ckplus_protocol_dataset(
            class_images,
            abnormal_classes=cli.abnormal_classes,
            normal_count=cli.normal_count,
            abnormal_count=cli.abnormal_count,
            augmentation_iterations=cli.augmentation_iterations,
            seed=seed,
        )
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
            x, y, seed=seed, test_size=cli.test_size, val_size=cli.val_size
        )
        train_loader, valid_loader, test_loader = make_loaders(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            hpo_args.train_batch_size,
            hpo_args.test_batch_size,
        )
        print(
            f"seed={seed}, normal_pool={sample_meta['normal_pool']}, "
            f"abnormal_pool={sample_meta['abnormal_pool']}, "
            f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
        )

        for budget in cli.budgets:
            for method in cli.methods:
                if (seed, budget, method) in done:
                    print(f"skipping existing seed={seed} budget={budget} method={method}")
                    continue

                simulation = f"seed_{seed}_b{budget}_{method.lower()}"
                set_seed(seed)
                print(f"running seed={seed} budget={budget} method={method}", flush=True)
                sync_device(device)
                start = time.perf_counter()
                if method == "Grid":
                    fit_candidate_search(
                        method,
                        grid_candidates(budget),
                        hpo_args,
                        train_loader,
                        valid_loader,
                        output_dir,
                        device,
                        simulation,
                    )
                elif method == "Random":
                    fit_candidate_search(
                        method,
                        random_candidates(budget),
                        hpo_args,
                        train_loader,
                        valid_loader,
                        output_dir,
                        device,
                        simulation,
                    )
                else:
                    fit_existing_search(
                        method,
                        budget,
                        hpo_args,
                        train_loader,
                        valid_loader,
                        output_dir,
                        device,
                        simulation,
                        cli,
                    )
                sync_device(device)
                elapsed = time.perf_counter() - start
                metrics = evaluate_method(method, hpo_args, test_loader, output_dir, device, simulation)
                row = {
                    "seed": seed,
                    "budget": budget,
                    "method": method,
                    "elapsed_sec": elapsed,
                    "lr": metrics["lr"],
                    "nu": metrics["nu"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "AUC": metrics["auc"],
                    "device": str(device),
                    "n_train": len(y_train),
                    "n_train_normal": int((y_train == 0).sum()),
                    "n_train_abnormal": int((y_train == 1).sum()),
                    "n_val": len(y_val),
                    "n_val_normal": int((y_val == 0).sum()),
                    "n_val_abnormal": int((y_val == 1).sum()),
                    "n_test": len(y_test),
                    "n_test_normal": int((y_test == 0).sum()),
                    "n_test_abnormal": int((y_test == 1).sum()),
                }
                append_budget_row(csv_path, row)
                print(
                    f"{method} B={budget}: elapsed={elapsed:.2f}s, "
                    f"Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, AUC={row['AUC']:.4f}"
                )
                summarize(csv_path, summary_path)


if __name__ == "__main__":
    main()
