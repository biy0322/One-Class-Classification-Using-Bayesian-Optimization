#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

OCC_SITE_PACKAGES = Path.home() / "anaconda3" / "envs" / "OCC" / "Lib" / "site-packages"
if OCC_SITE_PACKAGES.exists() and str(OCC_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(OCC_SITE_PACKAGES))

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from network.evaluation import eval as svdd_eval
from network.hopt import Bayesian, BOHB_hopt, GridSearch, Hyperband_hopt, RandomSearch
from network.network import LeNet5


METHODS = ("Grid", "Random", "Bayes", "Hyperband", "BOHB")
NORMAL_CLASS = "happy"
DEFAULT_ABNORMAL_CLASSES = ("anger", "disgust", "fear", "sadness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CK+ Deep SVDD HPO wall-clock time with happy as normal."
    )
    parser.add_argument("--data-root", default="data/CK", help="CK+ class-folder root.")
    parser.add_argument("--output-dir", default="results_wallclock", help="Output directory.")
    parser.add_argument("--seed-file", default="seed_list.txt", help="Seed list file.")
    parser.add_argument("--max-seeds", type=int, default=1, help="Number of seeds to run.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to evaluate.",
    )
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-epoch", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=5e-7)
    parser.add_argument("--bo-iter", type=int, default=25)
    parser.add_argument("--bo-kappa", type=float, default=15.0)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-resource", type=int, default=30)
    parser.add_argument("--min-resource", type=int, default=1)
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--grid-levels", type=int, default=5)
    parser.add_argument("--random-levels", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--abnormal-classes",
        nargs="+",
        default=list(DEFAULT_ABNORMAL_CLASSES),
        help="Emotion folders treated as abnormal. Contempt/surprise are excluded by default.",
    )
    parser.add_argument("--normal-count", type=int, default=1140)
    parser.add_argument("--abnormal-count", type=int, default=60)
    parser.add_argument("--augmentation-iterations", type=int, default=6)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for timing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows already present in the result CSV for the same seed and method.",
    )
    parser.add_argument(
        "--quiet-summary",
        action="store_true",
        help="Write summary CSV without printing the full table after each method.",
    )
    return parser.parse_args()


def make_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        num_epochs=cli.num_epochs,
        weight_decay=cli.weight_decay,
        train_batch_size=cli.train_batch_size,
        test_batch_size=cli.test_batch_size,
        latent_dim=cli.latent_dim,
        normal_class=0,
        early_stopping_epoch=cli.early_stopping_epoch,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_seeds(seed_file: Path, max_seeds: int) -> list[int]:
    seeds = []
    with seed_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(int(line))
    return seeds[:max_seeds]


def load_class_images(data_root: Path) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    class_images: dict[str, list[np.ndarray]] = {}
    class_counts: dict[str, int] = {}

    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        files = [
            p
            for p in sorted(class_dir.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
        images = []
        for file_path in files:
            image = cv2.imread(str(file_path))
            if image is None:
                continue
            images.append(image)
        class_images[class_name] = images
        class_counts[class_name] = len(images)

    if not class_images:
        raise RuntimeError(f"No image files were found under {data_root}")

    return class_images, class_counts


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_normal_images(
    images: list[np.ndarray],
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    augmented = [image.copy() for image in images]
    d_list = [20, 20, 30, 30, 40, 40]
    p_list = [0.5, 0.7, 0.5, 0.7, 0.5, 0.7]
    rng = np.random.default_rng(seed)

    for i in range(iterations):
        d = d_list[i % len(d_list)]
        p = p_list[i % len(p_list)]
        for image in images:
            angle = rng.uniform(-d, d)
            transformed = rotate_image(image, angle)
            if rng.random() < p:
                transformed = cv2.flip(transformed, 1)
            augmented.append(transformed)
    return augmented


def make_ckplus_protocol_dataset(
    class_images: dict[str, list[np.ndarray]],
    abnormal_classes: list[str],
    normal_count: int,
    abnormal_count: int,
    augmentation_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    missing = [name for name in [NORMAL_CLASS, *abnormal_classes] if name not in class_images]
    if missing:
        raise RuntimeError(f"Missing required CK+ class folders: {missing}")

    normal_pool = augment_normal_images(
        class_images[NORMAL_CLASS],
        iterations=augmentation_iterations,
        seed=seed,
    )
    abnormal_pool = [
        image
        for class_name in abnormal_classes
        for image in class_images[class_name]
    ]

    if len(normal_pool) < normal_count:
        raise RuntimeError(
            f"Not enough augmented normal images: requested {normal_count}, got {len(normal_pool)}."
        )
    if len(abnormal_pool) < abnormal_count:
        raise RuntimeError(
            f"Not enough abnormal images: requested {abnormal_count}, got {len(abnormal_pool)}."
        )

    rng = np.random.default_rng(seed)
    normal_idx = rng.choice(len(normal_pool), size=normal_count, replace=False)
    abnormal_idx = rng.choice(len(abnormal_pool), size=abnormal_count, replace=False)

    selected_normal = [normal_pool[i] for i in normal_idx]
    selected_abnormal = [abnormal_pool[i] for i in abnormal_idx]
    x = np.asarray(selected_normal + selected_abnormal)
    y = np.asarray([0] * normal_count + [1] * abnormal_count, dtype=np.int64)

    order = rng.permutation(len(y))
    meta = {
        "normal_pool": len(normal_pool),
        "abnormal_pool": len(abnormal_pool),
        "selected_normal": normal_count,
        "selected_abnormal": abnormal_count,
    }
    return x[order], y[order], meta


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=test_size, random_state=seed, shuffle=True
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, stratify=y_train, test_size=val_size, random_state=seed, shuffle=True
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def global_contrast_normalization(x: torch.Tensor, scale: str = "l1") -> torch.Tensor:
    x = x - torch.mean(x)
    if scale == "l1":
        x_scale = torch.mean(torch.abs(x))
    elif scale == "l2":
        x_scale = torch.sqrt(torch.sum(x ** 2)) / int(np.prod(x.shape))
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    return x / torch.clamp(x_scale, min=1e-12)


class CKPlusArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, scale: str = "l1"):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = cv2.resize(self.images[index], (32, 32), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        tensor = global_contrast_normalization(tensor, scale=self.scale)
        target = int(self.labels[index] == 1)
        return tensor, target, index


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_batch_size: int,
    test_batch_size: int,
):
    train_full = CKPlusArrayDataset(x_train, y_train, scale="l1")
    normal_idx = np.where(y_train == 0)[0].tolist()
    train_set = Subset(train_full, normal_idx)
    valid_set = CKPlusArrayDataset(x_val, y_val, scale="l1")
    test_set = CKPlusArrayDataset(x_test, y_test, scale="l1")
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=test_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader


def ensure_method_dirs(output_dir: Path, method: str, simulation: str) -> Path:
    method_dir = output_dir / "Model" / method
    (method_dir / "best_model" / simulation).mkdir(parents=True, exist_ok=True)
    return method_dir


def apply_budget_overrides(search, cli: argparse.Namespace, method: str) -> None:
    if method == "Grid" and cli.grid_levels != 5:
        search.param_grid = {
            "lr": list(np.linspace(0.0001, 0.01, cli.grid_levels)),
            "nu": list(np.linspace(0.001, 0.9999, cli.grid_levels)),
        }
    if method == "Random" and cli.random_levels != 5:
        random.seed(1500)
        search.param_random = {
            "lr": [random.uniform(0.0001, 0.01) for _ in range(cli.random_levels)],
            "nu": [random.uniform(0.001, 0.9999) for _ in range(cli.random_levels)],
        }


def fit_method(
    method: str,
    hpo_args: SimpleNamespace,
    train_loader,
    valid_loader,
    output_dir: Path,
    device: torch.device,
    simulation: str,
    cli: argparse.Namespace,
) -> None:
    method_dir = ensure_method_dirs(output_dir, method, simulation)
    path = str(method_dir) + os.sep
    early_stop = hpo_args.early_stopping_epoch

    if method == "Grid":
        search = GridSearch(
            hpo_args,
            train_loader,
            valid_loader,
            path=path,
            device=device,
            objective="soft-boundary",
            f_beta_param=1,
        )
        apply_budget_overrides(search, cli, method)
        search.fit(early_stop, simulation)
    elif method == "Random":
        search = RandomSearch(
            hpo_args,
            train_loader,
            valid_loader,
            path=path,
            device=device,
            objective="soft-boundary",
            f_beta_param=1,
        )
        apply_budget_overrides(search, cli, method)
        search.fit(early_stop, simulation)
    elif method == "Bayes":
        search = Bayesian(
            hpo_args,
            train_loader,
            valid_loader,
            path=path,
            device=device,
            name=simulation,
            early_stopping_epochs=early_stop,
            objective="soft-boundary",
            n_iter=cli.bo_iter,
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
            n_trials=cli.n_trials,
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
            n_trials=cli.n_trials,
            max_resource=cli.max_resource,
            min_resource=cli.min_resource,
            reduction_factor=cli.reduction_factor,
        )
        search.fit(simulation)
    else:
        raise ValueError(f"Unsupported method: {method}")


def evaluate_method(
    method: str,
    hpo_args: SimpleNamespace,
    test_loader,
    output_dir: Path,
    device: torch.device,
    simulation: str,
) -> dict[str, float]:
    model_file = output_dir / "Model" / method / "best_model" / simulation / f"{simulation}_result.pt"
    model_dict = torch.load(model_file, map_location=device, weights_only=False)
    net = LeNet5(hpo_args.latent_dim)
    net.load_state_dict(model_dict["net"])
    net.to(device)
    labels, scores = svdd_eval(net, model_dict["c"], model_dict["R"], "soft-boundary", test_loader, device)
    preds = np.asarray([0 if score <= 0 else 1 for score in scores])

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = float("nan")

    return {
        "lr": float(model_dict["lr"]),
        "nu": float(model_dict["nu"]),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": float(auc),
    }


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seed",
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
        writer.writerows(rows)


def existing_pairs(csv_path: Path) -> set[tuple[int, str]]:
    if not csv_path.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    return set(zip(df["seed"].astype(int), df["method"].astype(str)))


def summarize(csv_path: Path, summary_path: Path, print_table: bool = True) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    grouped = df.groupby("method").agg(
        elapsed_sec_mean=("elapsed_sec", "mean"),
        elapsed_sec_sem=("elapsed_sec", "sem"),
        Recall_mean=("Recall", "mean"),
        Recall_sem=("Recall", "sem"),
        F1_mean=("F1", "mean"),
        F1_sem=("F1", "sem"),
        AUC_mean=("AUC", "mean"),
        AUC_sem=("AUC", "sem"),
        lr_mean=("lr", "mean"),
        lr_std=("lr", "std"),
        nu_mean=("nu", "mean"),
        nu_std=("nu", "std"),
    )
    grouped.to_csv(summary_path)
    if print_table:
        print("\nsummary")
        print(grouped)


def main() -> None:
    cli = parse_args()
    base_dir = Path(__file__).resolve().parent
    data_root = (base_dir / cli.data_root).resolve()
    output_dir = (base_dir / cli.output_dir).resolve()
    seed_file = (base_dir / cli.seed_file).resolve()
    device = resolve_device(cli.device)
    hpo_args = make_args(cli)

    class_images, class_counts = load_class_images(data_root)
    seeds = read_seeds(seed_file, cli.max_seeds)
    csv_path = output_dir / "ckplus_wallclock_results.csv"
    summary_path = output_dir / "ckplus_wallclock_summary.csv"

    print(f"data_root={data_root}")
    print(f"class_counts={class_counts}")
    print(f"normal_class={NORMAL_CLASS}")
    print(f"abnormal_classes={cli.abnormal_classes}")
    print(
        f"sampling_protocol=normal {cli.normal_count}, "
        f"abnormal {cli.abnormal_count}, "
        f"augmentation_iterations {cli.augmentation_iterations}"
    )
    print(f"device={device}")
    print(f"seeds={seeds}")
    print(f"methods={cli.methods}")

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
            "sample "
            f"seed={seed}: normal_pool={sample_meta['normal_pool']}, "
            f"abnormal_pool={sample_meta['abnormal_pool']}, "
            f"selected_normal={sample_meta['selected_normal']}, "
            f"selected_abnormal={sample_meta['selected_abnormal']}"
        )
        print(
            "split "
            f"seed={seed}: train={len(y_train)} "
            f"(normal={(y_train == 0).sum()}, abnormal={(y_train == 1).sum()}), "
            f"val={len(y_val)} (normal={(y_val == 0).sum()}, abnormal={(y_val == 1).sum()}), "
            f"test={len(y_test)} (normal={(y_test == 0).sum()}, abnormal={(y_test == 1).sum()})"
        )

        done_pairs = existing_pairs(csv_path) if cli.skip_existing else set()
        for method in cli.methods:
            if (seed, method) in done_pairs:
                print(f"skipping existing {method} seed={seed}")
                continue
            simulation = f"seed_{seed}_{method.lower()}"
            set_seed(seed)
            print(f"running {method} seed={seed}", flush=True)
            sync_device(device)
            start = time.perf_counter()
            fit_method(method, hpo_args, train_loader, valid_loader, output_dir, device, simulation, cli)
            sync_device(device)
            elapsed = time.perf_counter() - start
            metrics = evaluate_method(method, hpo_args, test_loader, output_dir, device, simulation)
            row = {
                "seed": seed,
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
            print(
                f"{method}: elapsed={elapsed:.2f}s, "
                f"Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, AUC={row['AUC']:.4f}"
            )
            append_rows(csv_path, [row])
            summarize(csv_path, summary_path, print_table=not cli.quiet_summary)


if __name__ == "__main__":
    main()
