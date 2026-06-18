import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.metrics import f1_score, fbeta_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM


warnings.filterwarnings("ignore", category=ConvergenceWarning)

METHODS = ["GC", "RC", "BO", "HB", "BOHB"]
BO_KERNELS = ["rbf", "matern", "rational_quadratic", "linear"]
GAMMA_BOUNDS = (1e-4, 1.0)
NU_BOUNDS = (0.01, 0.50)
CATEGORICAL_COLS = [1, 2, 3]
LABEL_COL = 41
DIFFICULTY_COL = 42
ATTACK_GROUPS = {
    "dos": {
        "back",
        "land",
        "neptune",
        "pod",
        "smurf",
        "teardrop",
        "apache2",
        "mailbomb",
        "processtable",
        "udpstorm",
    },
    "probe": {
        "ipsweep",
        "mscan",
        "nmap",
        "portsweep",
        "saint",
        "satan",
    },
    "r2l": {
        "ftp_write",
        "guess_passwd",
        "httptunnel",
        "imap",
        "multihop",
        "named",
        "phf",
        "sendmail",
        "snmpgetattack",
        "snmpguess",
        "spy",
        "warezclient",
        "warezmaster",
        "worm",
        "xlock",
        "xsnoop",
    },
    "u2r": {
        "buffer_overflow",
        "loadmodule",
        "perl",
        "ps",
        "rootkit",
        "sqlattack",
        "xterm",
    },
}


def load_nsl_kdd(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    feature_cols = [c for c in train_df.columns if c not in (LABEL_COL, DIFFICULTY_COL)]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    train_y = (train_df[LABEL_COL] != "normal").astype(int).to_numpy()
    test_y = (test_df[LABEL_COL] != "normal").astype(int).to_numpy()

    return (
        train_df[feature_cols],
        train_y,
        test_df[feature_cols],
        test_y,
        numeric_cols,
        CATEGORICAL_COLS,
        train_df[LABEL_COL],
        test_df[LABEL_COL],
    )


def apply_target_attack_rate(x_raw, y, labels, target_attack_rate, seed, target_total_size=None):
    if target_attack_rate is None and target_total_size is None:
        return x_raw, y, labels
    if target_attack_rate is None:
        raise ValueError("--target-total-size requires --target-attack-rate.")
    if not 0.0 < target_attack_rate < 1.0:
        raise ValueError("--target-attack-rate must be between 0 and 1.")

    normal_idx = np.flatnonzero(y == 0)
    attack_idx = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)

    if target_total_size is not None:
        if target_total_size <= 1:
            raise ValueError("--target-total-size must be greater than 1.")
        target_attack_count = int(round(target_total_size * target_attack_rate))
        target_attack_count = max(1, target_attack_count)
        target_normal_count = int(target_total_size) - target_attack_count
        if target_normal_count > len(normal_idx) or target_attack_count > len(attack_idx):
            raise ValueError(
                "Requested target size exceeds available normal or attack observations."
            )
        sampled_normal_idx = rng.choice(normal_idx, size=target_normal_count, replace=False)
        sampled_attack_idx = rng.choice(attack_idx, size=target_attack_count, replace=False)
        selected_idx = np.concatenate([sampled_normal_idx, sampled_attack_idx])
    else:
        target_attack_count = int(round(len(normal_idx) * target_attack_rate / (1.0 - target_attack_rate)))
        target_attack_count = min(target_attack_count, len(attack_idx))
        if target_attack_count >= len(attack_idx):
            return x_raw, y, labels
        sampled_attack_idx = rng.choice(attack_idx, size=target_attack_count, replace=False)
        selected_idx = np.concatenate([normal_idx, sampled_attack_idx])

    selected_idx.sort()
    return (
        x_raw.iloc[selected_idx].reset_index(drop=True),
        y[selected_idx],
        labels.iloc[selected_idx].reset_index(drop=True),
    )


def filter_attack_group(x_raw, labels, attack_group):
    if attack_group == "all":
        y = (labels != "normal").astype(int).to_numpy()
        return x_raw, y, labels
    selected_attacks = ATTACK_GROUPS[attack_group]
    keep_mask = (labels == "normal") | labels.isin(selected_attacks)
    filtered_x = x_raw.loc[keep_mask].reset_index(drop=True)
    filtered_labels = labels.loc[keep_mask].reset_index(drop=True)
    filtered_y = (filtered_labels != "normal").astype(int).to_numpy()
    return filtered_x, filtered_y, filtered_labels


def make_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def stratified_subsample_indices(y, max_samples, seed):
    if max_samples is None or len(y) <= max_samples:
        return np.arange(len(y))
    idx = np.arange(len(y))
    _, sample_idx = train_test_split(
        idx, test_size=max_samples, stratify=y, random_state=seed
    )
    return sample_idx


def split_and_transform(
    train_x_raw,
    train_y,
    test_x_raw,
    test_y,
    numeric_cols,
    categorical_cols,
    seed,
    val_size,
    max_fit_normals,
    max_val_samples,
    transformed_feature_count=None,
):
    idx = np.arange(len(train_y))
    fit_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=train_y, random_state=seed
    )

    normal_fit_idx = fit_idx[train_y[fit_idx] == 0]
    rng = np.random.default_rng(seed)
    if len(normal_fit_idx) > max_fit_normals:
        normal_fit_idx = rng.choice(normal_fit_idx, size=max_fit_normals, replace=False)

    val_sub_idx = stratified_subsample_indices(train_y[val_idx], max_val_samples, seed + 5000)
    val_idx = val_idx[val_sub_idx]

    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(train_x_raw.iloc[normal_fit_idx])

    x_fit = preprocessor.transform(train_x_raw.iloc[normal_fit_idx]).astype(np.float32)
    x_val = preprocessor.transform(train_x_raw.iloc[val_idx]).astype(np.float32)
    x_test = preprocessor.transform(test_x_raw).astype(np.float32)

    original_transformed_dim = int(x_fit.shape[1])
    selected_transformed_dim = original_transformed_dim
    if transformed_feature_count is not None:
        if transformed_feature_count <= 0:
            raise ValueError("--transformed-feature-count must be positive.")
        selected_transformed_dim = min(int(transformed_feature_count), original_transformed_dim)
        variances = np.var(x_fit, axis=0)
        selected_cols = np.argsort(variances)[::-1][:selected_transformed_dim]
        selected_cols.sort()
        x_fit = x_fit[:, selected_cols]
        x_val = x_val[:, selected_cols]
        x_test = x_test[:, selected_cols]

    return x_fit, x_val, train_y[val_idx], x_test, test_y, original_transformed_dim, selected_transformed_dim


def predict_ocsvm(model, x):
    raw = model.predict(x)
    return np.where(raw == -1, 1, 0)


def metric_bundle(y_true, y_pred):
    return {
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F-1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC": roc_auc_score(y_true, y_pred),
    }


def validation_score(
    x_fit,
    x_val,
    y_val,
    gamma,
    nu,
    subset_seed=None,
    subset_frac=1.0,
    validation_metric="f1",
    validation_beta=1.0,
    validation_average="macro",
    return_timing=False,
):
    total_start = time.perf_counter()
    subset_start = time.perf_counter()
    x_fit_use = x_fit
    if subset_frac < 1.0:
        rng = np.random.default_rng(subset_seed)
        n = max(100, int(len(x_fit) * subset_frac))
        n = min(n, len(x_fit))
        idx = rng.choice(len(x_fit), size=n, replace=False)
        x_fit_use = x_fit[idx]
    subset_time = time.perf_counter() - subset_start

    model = OneClassSVM(kernel="rbf", gamma=float(gamma), nu=float(nu))
    fit_start = time.perf_counter()
    model.fit(x_fit_use)
    fit_time = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    pred = predict_ocsvm(model, x_val)
    predict_time = time.perf_counter() - predict_start
    score_start = time.perf_counter()
    if validation_metric == "f_beta":
        score = fbeta_score(
            y_val,
            pred,
            beta=float(validation_beta),
            average=validation_average,
            pos_label=1,
            zero_division=0,
        )
    else:
        score = f1_score(y_val, pred, average="macro", zero_division=0)
    score_time = time.perf_counter() - score_start
    if not return_timing:
        return score
    return score, {
        "subset_time_sec": subset_time,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "score_time_sec": score_time,
        "eval_duration_sec": time.perf_counter() - total_start,
    }


def make_gp_kernel(name):
    if name in {"rbf", "squared_exponential", "se"}:
        base = RBF(length_scale=[1.0, 0.1], length_scale_bounds=(1e-2, 1e2))
    elif name == "matern":
        base = Matern(length_scale=[1.0, 0.1], length_scale_bounds=(1e-2, 1e2), nu=2.5)
    elif name in {"rational_quadratic", "rq"}:
        base = RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2),
        )
    elif name == "linear":
        base = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
    else:
        raise ValueError(f"Unsupported BO kernel: {name}")
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * base
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )


def append_history(
    history,
    start_time,
    eval_index,
    gamma,
    nu,
    score,
    trial_number=None,
    resource_step=1,
    full_eval=True,
    status="complete",
    eval_duration_sec=0.0,
    subset_time_sec=0.0,
    fit_time_sec=0.0,
    predict_time_sec=0.0,
    score_time_sec=0.0,
    surrogate_fit_time_sec=0.0,
    acquisition_time_sec=0.0,
):
    previous = [row["best_so_far"] for row in history if row.get("full_eval", True)]
    best_so_far = max(previous, default=-np.inf)
    if full_eval:
        best_so_far = max(best_so_far, score)
    history.append(
        {
            "eval_index": int(eval_index),
            "trial_number": trial_number,
            "resource_step": int(resource_step),
            "gamma": float(gamma),
            "nu": float(nu),
            "score": float(score),
            "best_so_far": float(best_so_far),
            "elapsed_sec": time.perf_counter() - start_time,
            "eval_duration_sec": float(eval_duration_sec),
            "subset_time_sec": float(subset_time_sec),
            "fit_time_sec": float(fit_time_sec),
            "predict_time_sec": float(predict_time_sec),
            "score_time_sec": float(score_time_sec),
            "surrogate_fit_time_sec": float(surrogate_fit_time_sec),
            "acquisition_time_sec": float(acquisition_time_sec),
            "full_eval": bool(full_eval),
            "status": status,
        }
    )


def summarize_history(history):
    full_history = [row for row in history if row.get("full_eval", True)]
    if not full_history:
        raise ValueError("No full validation evaluations were recorded.")

    best_score = max(row["score"] for row in full_history)
    first_best = next(row for row in full_history if row["score"] == best_score)
    trial_number = first_best.get("trial_number")
    if trial_number is None or (isinstance(trial_number, float) and np.isnan(trial_number)):
        configs_to_best = first_best["eval_index"]
    else:
        configs_to_best = int(trial_number) + 1

    total_time_sec = float(max(row["elapsed_sec"] for row in history))
    total_evaluation_time_sec = float(sum(row.get("eval_duration_sec", 0.0) for row in history))
    total_subset_time_sec = float(sum(row.get("subset_time_sec", 0.0) for row in history))
    total_fit_time_sec = float(sum(row.get("fit_time_sec", 0.0) for row in history))
    total_predict_time_sec = float(sum(row.get("predict_time_sec", 0.0) for row in history))
    total_score_time_sec = float(sum(row.get("score_time_sec", 0.0) for row in history))
    total_surrogate_fit_time_sec = float(sum(row.get("surrogate_fit_time_sec", 0.0) for row in history))
    total_acquisition_time_sec = float(sum(row.get("acquisition_time_sec", 0.0) for row in history))
    optimization_overhead_sec = max(0.0, total_time_sec - total_evaluation_time_sec)
    trial_numbers = [
        row.get("trial_number")
        for row in history
        if row.get("trial_number") is not None
        and not (isinstance(row.get("trial_number"), float) and np.isnan(row.get("trial_number")))
    ]
    total_configurations = int(max(trial_numbers) + 1) if trial_numbers else int(max(row["eval_index"] for row in history))

    return {
        "best_validation_score": best_score,
        "configs_to_best": int(configs_to_best),
        "observed_evals_to_best": int(first_best["eval_index"]),
        "time_to_best_sec": float(first_best["elapsed_sec"]),
        "total_configurations": total_configurations,
        "full_validation_evals": int(sum(1 for row in history if row.get("full_eval", True))),
        "total_observed_evals": int(max(row["eval_index"] for row in history)),
        "total_time_sec": total_time_sec,
        "total_evaluation_time_sec": total_evaluation_time_sec,
        "total_subset_time_sec": total_subset_time_sec,
        "total_fit_time_sec": total_fit_time_sec,
        "total_predict_time_sec": total_predict_time_sec,
        "total_score_time_sec": total_score_time_sec,
        "total_surrogate_fit_time_sec": total_surrogate_fit_time_sec,
        "total_acquisition_time_sec": total_acquisition_time_sec,
        "optimization_overhead_sec": optimization_overhead_sec,
        "optimization_overhead_pct": 100.0 * optimization_overhead_sec / total_time_sec if total_time_sec > 0 else 0.0,
    }


def fit_and_test(x_fit, x_test, y_test, gamma, nu):
    model = OneClassSVM(kernel="rbf", gamma=float(gamma), nu=float(nu))
    model.fit(x_fit)
    pred = predict_ocsvm(model, x_test)
    return metric_bundle(y_test, pred)


def grid_search(
    x_fit,
    x_val,
    y_val,
    grid_size,
    validation_metric="f1",
    validation_beta=1.0,
    validation_average="macro",
):
    gammas = np.logspace(math.log10(GAMMA_BOUNDS[0]), math.log10(GAMMA_BOUNDS[1]), grid_size)
    nus = np.linspace(NU_BOUNDS[0], NU_BOUNDS[1], grid_size)
    best = (-np.inf, None)
    history = []
    start_time = time.perf_counter()
    eval_index = 0
    for gamma in gammas:
        for nu in nus:
            eval_index += 1
            score, timing = validation_score(
                x_fit,
                x_val,
                y_val,
                gamma,
                nu,
                validation_metric=validation_metric,
                validation_beta=validation_beta,
                validation_average=validation_average,
                return_timing=True,
            )
            if score > best[0]:
                best = (score, {"gamma": gamma, "nu": nu})
            append_history(
                history,
                start_time,
                eval_index,
                gamma,
                nu,
                score,
                trial_number=eval_index - 1,
                **timing,
            )
    return best[1], history


def random_search(
    x_fit,
    x_val,
    y_val,
    rng,
    n_trials,
    validation_metric="f1",
    validation_beta=1.0,
    validation_average="macro",
):
    best = (-np.inf, None)
    history = []
    start_time = time.perf_counter()
    for trial in range(n_trials):
        gamma = 10 ** rng.uniform(math.log10(GAMMA_BOUNDS[0]), math.log10(GAMMA_BOUNDS[1]))
        nu = rng.uniform(*NU_BOUNDS)
        score, timing = validation_score(
            x_fit,
            x_val,
            y_val,
            gamma,
            nu,
            validation_metric=validation_metric,
            validation_beta=validation_beta,
            validation_average=validation_average,
            return_timing=True,
        )
        if score > best[0]:
            best = (score, {"gamma": gamma, "nu": nu})
        append_history(
            history,
            start_time,
            trial + 1,
            gamma,
            nu,
            score,
            trial_number=trial,
            **timing,
        )
    return best[1], history


def bayes_search(
    x_fit,
    x_val,
    y_val,
    rng,
    n_trials,
    n_init=5,
    kappa=1.96,
    bo_kernel="rbf",
    validation_metric="f1",
    validation_beta=1.0,
    validation_average="macro",
):
    observations_x = []
    observations_y = []
    history = []
    start_time = time.perf_counter()

    def sample_params():
        log_gamma = rng.uniform(math.log10(GAMMA_BOUNDS[0]), math.log10(GAMMA_BOUNDS[1]))
        nu = rng.uniform(*NU_BOUNDS)
        return log_gamma, nu

    def evaluate(log_gamma, nu):
        score, timing = validation_score(
            x_fit,
            x_val,
            y_val,
            10**log_gamma,
            nu,
            validation_metric=validation_metric,
            validation_beta=validation_beta,
            validation_average=validation_average,
            return_timing=True,
        )
        return score, timing

    for _ in range(min(n_init, n_trials)):
        log_gamma, nu = sample_params()
        observations_x.append([log_gamma, nu])
        score, timing = evaluate(log_gamma, nu)
        observations_y.append(score)
        append_history(
            history,
            start_time,
            len(observations_y),
            10**log_gamma,
            nu,
            score,
            trial_number=len(observations_y) - 1,
            **timing,
        )

    while len(observations_y) < n_trials:
        kernel = make_gp_kernel(bo_kernel)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=int(rng.integers(1_000_000)),
            n_restarts_optimizer=1,
        )
        x_obs = np.asarray(observations_x)
        y_obs = np.asarray(observations_y)
        surrogate_start = time.perf_counter()
        gp.fit(x_obs, y_obs)
        surrogate_fit_time = time.perf_counter() - surrogate_start

        acquisition_start = time.perf_counter()
        candidates = np.column_stack(
            [
                rng.uniform(math.log10(GAMMA_BOUNDS[0]), math.log10(GAMMA_BOUNDS[1]), 256),
                rng.uniform(NU_BOUNDS[0], NU_BOUNDS[1], 256),
            ]
        )
        mu, std = gp.predict(candidates, return_std=True)
        log_gamma, nu = candidates[int(np.argmax(mu + kappa * std))]
        acquisition_time = time.perf_counter() - acquisition_start
        observations_x.append([float(log_gamma), float(nu)])
        score, timing = evaluate(log_gamma, nu)
        observations_y.append(score)
        append_history(
            history,
            start_time,
            len(observations_y),
            10**log_gamma,
            nu,
            score,
            trial_number=len(observations_y) - 1,
            surrogate_fit_time_sec=surrogate_fit_time,
            acquisition_time_sec=acquisition_time,
            **timing,
        )

    best_idx = int(np.argmax(observations_y))
    best_log_gamma, best_nu = observations_x[best_idx]
    return {"gamma": 10**best_log_gamma, "nu": best_nu}, history


def hyperband_search(
    x_fit,
    x_val,
    y_val,
    seed,
    n_trials,
    sampler_name,
    validation_metric="f1",
    validation_beta=1.0,
    validation_average="macro",
):
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(sampler_name)

    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)
    history = []
    start_time = time.perf_counter()

    def objective(trial):
        gamma = trial.suggest_float("gamma", GAMMA_BOUNDS[0], GAMMA_BOUNDS[1], log=True)
        nu = trial.suggest_float("nu", NU_BOUNDS[0], NU_BOUNDS[1])
        last = None
        for resource in range(1, 4):
            last, timing = validation_score(
                x_fit,
                x_val,
                y_val,
                gamma,
                nu,
                subset_seed=seed + trial.number * 10 + resource,
                subset_frac=resource / 3.0,
                validation_metric=validation_metric,
                validation_beta=validation_beta,
                validation_average=validation_average,
                return_timing=True,
            )
            trial.report(last, step=resource)
            append_history(
                history,
                start_time,
                len(history) + 1,
                gamma,
                nu,
                last,
                trial_number=trial.number,
                resource_step=resource,
                full_eval=(resource == 3),
                status="observed",
                **timing,
            )
            if trial.should_prune():
                history[-1]["status"] = "pruned"
                raise optuna.TrialPruned()
        return last

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, catch=(ValueError,))
    return dict(study.best_params), history


def run_experiment(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_n_trials = args.grid_size * args.grid_size if args.equal_budget_from_grid else args.n_trials
    selected_methods = list(dict.fromkeys(args.methods))

    (
        train_x_raw,
        train_y,
        test_x_raw,
        test_y,
        numeric_cols,
        categorical_cols,
        train_labels,
        test_labels,
    ) = load_nsl_kdd(args.train, args.test)

    train_x_raw, train_y, train_labels = filter_attack_group(
        train_x_raw,
        train_labels,
        args.attack_group,
    )
    test_x_raw, test_y, test_labels = filter_attack_group(
        test_x_raw,
        test_labels,
        args.attack_group,
    )

    train_x_raw, train_y, train_labels = apply_target_attack_rate(
        train_x_raw,
        train_y,
        train_labels,
        args.target_attack_rate,
        args.seed + 100_000,
        args.target_train_size,
    )
    test_x_raw, test_y, test_labels = apply_target_attack_rate(
        test_x_raw,
        test_y,
        test_labels,
        args.target_attack_rate,
        args.seed + 200_000,
        args.target_test_size,
    )

    meta = {
        "train_file": str(args.train),
        "test_file": str(args.test),
        "n_train": int(len(train_y)),
        "n_train_normal": int((train_y == 0).sum()),
        "n_train_attack": int((train_y == 1).sum()),
        "train_attack_rate": float((train_y == 1).mean()),
        "n_test": int(len(test_y)),
        "n_test_normal": int((test_y == 0).sum()),
        "n_test_attack": int((test_y == 1).sum()),
        "test_attack_rate": float((test_y == 1).mean()),
        "n_repeats": args.n_repeats,
        "n_trials": effective_n_trials,
        "grid_size": args.grid_size,
        "grid_budget": args.grid_size * args.grid_size,
        "equal_budget_from_grid": bool(args.equal_budget_from_grid),
        "max_fit_normals": args.max_fit_normals,
        "max_val_samples": args.max_val_samples,
        "methods": ",".join(selected_methods),
        "validation_metric": args.validation_metric,
        "validation_beta": args.validation_beta,
        "validation_average": args.validation_average,
        "bo_kernel": args.bo_kernel,
        "performance_only": bool(args.performance_only),
        "attack_group": args.attack_group,
        "target_attack_rate": args.target_attack_rate if args.target_attack_rate is not None else "",
        "target_train_size": args.target_train_size if args.target_train_size is not None else "",
        "target_test_size": args.target_test_size if args.target_test_size is not None else "",
        "transformed_feature_count": args.transformed_feature_count if args.transformed_feature_count is not None else "",
    }
    print("metadata", meta, flush=True)
    print("top_train_attacks", train_labels[train_labels != "normal"].value_counts().head(10).to_dict())
    print("top_test_attacks", test_labels[test_labels != "normal"].value_counts().head(10).to_dict())

    rows = []
    history_rows = []
    time_rows = []
    for rep in range(args.n_repeats):
        seed = args.seed + rep
        (
            x_fit,
            x_val,
            y_val,
            x_test,
            y_test,
            original_transformed_dim,
            selected_transformed_dim,
        ) = split_and_transform(
            train_x_raw,
            train_y,
            test_x_raw,
            test_y,
            numeric_cols,
            categorical_cols,
            seed=seed,
            val_size=args.val_size,
            max_fit_normals=args.max_fit_normals,
            max_val_samples=args.max_val_samples,
            transformed_feature_count=args.transformed_feature_count,
        )
        rng = np.random.default_rng(seed)
        print(
            f"repeat {rep + 1}/{args.n_repeats}: fit_normals={len(x_fit)}, "
            f"val={len(y_val)}, val_attacks={int(y_val.sum())}, "
            f"test={len(y_test)}, test_attacks={int(y_test.sum())}, "
            f"features={selected_transformed_dim}/{original_transformed_dim}",
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
                bo_kernel=args.bo_kernel,
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
            search = searches[method]
            start = time.time()
            params, history = search()
            metrics = fit_and_test(x_fit, x_test, y_test, **params)
            elapsed = time.time() - start
            row = {
                "repeat": rep,
                "method": method,
                "validation_metric": args.validation_metric,
                "validation_beta": args.validation_beta,
                "validation_average": args.validation_average,
                "bo_kernel": args.bo_kernel if method == "BO" else "",
                "transformed_features": selected_transformed_dim,
                "original_transformed_features": original_transformed_dim,
                "gamma": params["gamma"],
                "nu": params["nu"],
                **metrics,
            }
            if not args.performance_only:
                time_summary = summarize_history(history)
                row.update(
                    {
                        "elapsed_sec": elapsed,
                        "best_validation_score": time_summary["best_validation_score"],
                        "configs_to_best": time_summary["configs_to_best"],
                        "observed_evals_to_best": time_summary["observed_evals_to_best"],
                        "time_to_best_sec": time_summary["time_to_best_sec"],
                        "total_observed_evals": time_summary["total_observed_evals"],
                        "total_time_sec": time_summary["total_time_sec"],
                        "total_evaluation_time_sec": time_summary["total_evaluation_time_sec"],
                        "total_fit_time_sec": time_summary["total_fit_time_sec"],
                        "total_predict_time_sec": time_summary["total_predict_time_sec"],
                        "total_score_time_sec": time_summary["total_score_time_sec"],
                        "total_surrogate_fit_time_sec": time_summary["total_surrogate_fit_time_sec"],
                        "total_acquisition_time_sec": time_summary["total_acquisition_time_sec"],
                        "optimization_overhead_sec": time_summary["optimization_overhead_sec"],
                    }
                )
            rows.append(row)
            if not args.performance_only:
                time_rows.append(
                    {
                        "repeat": rep,
                        "method": method,
                        "validation_metric": args.validation_metric,
                        "validation_beta": args.validation_beta,
                        "validation_average": args.validation_average,
                        "bo_kernel": args.bo_kernel if method == "BO" else "",
                        "transformed_features": selected_transformed_dim,
                        "original_transformed_features": original_transformed_dim,
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
                            "bo_kernel": args.bo_kernel if method == "BO" else "",
                            "transformed_features": selected_transformed_dim,
                            "original_transformed_features": original_transformed_dim,
                            **h,
                        }
                    )
                timing_text = (
                    f", time_to_best={time_summary['time_to_best_sec']:.2f}s, "
                    f"total={elapsed:.2f}s"
                )
            else:
                timing_text = ""
            print(
                f"  {method}: gamma={params['gamma']:.6g}, nu={params['nu']:.4f}, "
                f"Recall={metrics['Recall']:.4f}, F-1={metrics['F-1']:.4f}, "
                f"AUC={metrics['AUC']:.4f}"
                f"{timing_text}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "nsl_kdd_ocsvm_repeated_results.csv", index=False)

    summary = (
        results.groupby("method")[["Recall", "F-1", "AUC"]]
        .agg(["mean", "sem"])
        .reindex(selected_methods)
    )
    summary.to_csv(args.output_dir / "nsl_kdd_ocsvm_summary.csv")
    if not args.performance_only:
        pd.DataFrame(history_rows).to_csv(args.output_dir / "nsl_kdd_ocsvm_search_history.csv", index=False)
        time_to_best = pd.DataFrame(time_rows)
        time_to_best.to_csv(args.output_dir / "nsl_kdd_ocsvm_time_to_best.csv", index=False)
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
                    "total_subset_time_sec",
                    "total_fit_time_sec",
                    "total_predict_time_sec",
                    "total_score_time_sec",
                    "total_surrogate_fit_time_sec",
                    "total_acquisition_time_sec",
                    "optimization_overhead_sec",
                    "optimization_overhead_pct",
                ]
            ]
            .agg(["mean", "sem"])
            .reindex(selected_methods)
        )
        time_summary.to_csv(args.output_dir / "nsl_kdd_ocsvm_time_summary.csv")
    pd.DataFrame([meta]).to_csv(args.output_dir / "nsl_kdd_ocsvm_metadata.csv", index=False)

    print("\nsummary")
    print(summary)
    if not args.performance_only:
        print("\ntime summary")
        print(time_summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data_candidates/NSL-KDD/KDDTrain+.txt"))
    parser.add_argument("--test", type=Path, default=Path("data_candidates/NSL-KDD/KDDTest+.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("NSL-KDD/results"))
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument(
        "--equal-budget-from-grid",
        action="store_true",
        help="Use grid_size^2 as the trial budget for Random, BO, HB, and BOHB.",
    )
    parser.add_argument("--max-fit-normals", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=6000)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument(
        "--attack-group",
        choices=["all", *ATTACK_GROUPS.keys()],
        default="all",
        help="Filter NSL-KDD to normal samples plus one attack category.",
    )
    parser.add_argument(
        "--target-attack-rate",
        type=float,
        default=None,
        help="Optionally subsample attack observations in train and test to this attack rate.",
    )
    parser.add_argument(
        "--target-train-size",
        type=int,
        default=None,
        help="Optionally subsample the training file to this total size after applying the attack rate.",
    )
    parser.add_argument(
        "--target-test-size",
        type=int,
        default=None,
        help="Optionally subsample the test file to this total size after applying the attack rate.",
    )
    parser.add_argument(
        "--transformed-feature-count",
        type=int,
        default=None,
        help="Use only the top-k preprocessed features ranked by variance on the normal fitting split.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Subset of HPO methods to run.",
    )
    parser.add_argument(
        "--validation-metric",
        choices=["f1", "f_beta"],
        default="f1",
        help="Validation objective used during hyperparameter search.",
    )
    parser.add_argument(
        "--validation-beta",
        type=float,
        default=1.0,
        help="Beta value for --validation-metric f_beta.",
    )
    parser.add_argument(
        "--validation-average",
        choices=["binary", "macro"],
        default="macro",
        help="Averaging mode for --validation-metric f_beta.",
    )
    parser.add_argument(
        "--bo-kernel",
        choices=BO_KERNELS,
        default="rbf",
        help="GP surrogate kernel for Bayesian optimization. RBF is the squared-exponential kernel.",
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Report only final test performance and omit search-history/timing outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
