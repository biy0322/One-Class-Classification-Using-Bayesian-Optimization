from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_forest_fire(data_file: Path | None = None):
    if data_file is None:
        data_file = REPO_ROOT / "ForestFire" / "data_manipulated.csv"
    df = pd.read_csv(data_file)
    x = df[["PREC", "RH", "WIND_SPEED", "TEMP"]].astype(float)
    y = (df["y"] == "pos").astype(int).to_numpy()
    return x, y, data_file


def load_dataset(name, data_root=None):
    if name != "forest_fire":
        raise ValueError(f"Unknown dataset: {name}")
    data_file = None
    if data_root is not None:
        data_root = Path(data_root)
        data_file = data_root if data_root.is_file() else data_root / "data_manipulated.csv"
    return load_forest_fire(data_file)


def make_split(x_raw, y, seed, test_size, val_size):
    normal_idx = np.flatnonzero(y == 0)
    abnormal_idx = np.flatnonzero(y == 1)
    normal_train_idx, normal_test_idx = train_test_split(
        normal_idx,
        test_size=test_size,
        random_state=seed,
    )
    abnormal_train_idx, abnormal_test_idx = train_test_split(
        abnormal_idx,
        test_size=test_size,
        random_state=seed,
    )
    train_idx = np.concatenate([normal_train_idx, abnormal_train_idx])
    test_idx = np.concatenate([normal_test_idx, abnormal_test_idx])

    fit_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        stratify=y[train_idx],
        random_state=seed + 1000,
    )

    search_fit_idx = fit_idx[y[fit_idx] == 0]
    final_fit_idx = train_idx[y[train_idx] == 0]

    search_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    search_preprocessor.fit(x_raw.iloc[search_fit_idx])
    x_search_fit = search_preprocessor.transform(x_raw.iloc[search_fit_idx]).astype(np.float32)
    x_val = search_preprocessor.transform(x_raw.iloc[val_idx]).astype(np.float32)

    final_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    final_preprocessor.fit(x_raw.iloc[final_fit_idx])
    x_final_fit = final_preprocessor.transform(x_raw.iloc[final_fit_idx]).astype(np.float32)
    x_test = final_preprocessor.transform(x_raw.iloc[test_idx]).astype(np.float32)

    return {
        "x_search_fit": x_search_fit,
        "x_val": x_val,
        "y_val": y[val_idx],
        "x_final_fit": x_final_fit,
        "x_test": x_test,
        "y_test": y[test_idx],
        "n_train_normal": int((y[train_idx] == 0).sum()),
        "n_train_abnormal": int((y[train_idx] == 1).sum()),
        "n_test_normal": int((y[test_idx] == 0).sum()),
        "n_test_abnormal": int((y[test_idx] == 1).sum()),
    }


def summarize_metrics(results, selected_methods):
    return (
        results.groupby("method")[["Recall", "F-1", "AUC"]]
        .agg(["mean", "sem"])
        .reindex(selected_methods)
    )