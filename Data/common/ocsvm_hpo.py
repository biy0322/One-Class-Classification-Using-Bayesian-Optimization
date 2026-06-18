import math
import time
import warnings

import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import f1_score, fbeta_score, recall_score, roc_auc_score
from sklearn.svm import OneClassSVM


warnings.filterwarnings("ignore", category=ConvergenceWarning)

METHODS = ["GC", "RC", "BO", "HB", "BOHB"]
GAMMA_BOUNDS = (1e-4, 1.0)
NU_BOUNDS = (0.01, 0.50)

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
):
    x_fit_use = x_fit
    if subset_frac < 1.0:
        rng = np.random.default_rng(subset_seed)
        n = max(30, int(len(x_fit) * subset_frac))
        n = min(n, len(x_fit))
        idx = rng.choice(len(x_fit), size=n, replace=False)
        x_fit_use = x_fit[idx]

    model = OneClassSVM(kernel="rbf", gamma=float(gamma), nu=float(nu))
    model.fit(x_fit_use)
    pred = predict_ocsvm(model, x_val)
    if validation_metric == "f_beta":
        return fbeta_score(
            y_val,
            pred,
            beta=float(validation_beta),
            average=validation_average,
            pos_label=1,
            zero_division=0,
        )
    return f1_score(y_val, pred, average="macro", zero_division=0)


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
            eval_start = time.perf_counter()
            score = validation_score(
                x_fit,
                x_val,
                y_val,
                gamma,
                nu,
                validation_metric=validation_metric,
                validation_beta=validation_beta,
                validation_average=validation_average,
            )
            eval_duration = time.perf_counter() - eval_start
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
                eval_duration_sec=eval_duration,
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
        eval_start = time.perf_counter()
        score = validation_score(
            x_fit,
            x_val,
            y_val,
            gamma,
            nu,
            validation_metric=validation_metric,
            validation_beta=validation_beta,
            validation_average=validation_average,
        )
        eval_duration = time.perf_counter() - eval_start
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
            eval_duration_sec=eval_duration,
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
        eval_start = time.perf_counter()
        score = validation_score(
            x_fit,
            x_val,
            y_val,
            10**log_gamma,
            nu,
            validation_metric=validation_metric,
            validation_beta=validation_beta,
            validation_average=validation_average,
        )
        return score, time.perf_counter() - eval_start

    for _ in range(min(n_init, n_trials)):
        log_gamma, nu = sample_params()
        observations_x.append([log_gamma, nu])
        score, eval_duration = evaluate(log_gamma, nu)
        observations_y.append(score)
        append_history(
            history,
            start_time,
            len(observations_y),
            10**log_gamma,
            nu,
            score,
            trial_number=len(observations_y) - 1,
            eval_duration_sec=eval_duration,
        )

    while len(observations_y) < n_trials:
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0, 0.1], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=int(rng.integers(1_000_000)),
            n_restarts_optimizer=1,
        )
        x_obs = np.asarray(observations_x)
        y_obs = np.asarray(observations_y)
        gp.fit(x_obs, y_obs)

        candidates = np.column_stack(
            [
                rng.uniform(math.log10(GAMMA_BOUNDS[0]), math.log10(GAMMA_BOUNDS[1]), 256),
                rng.uniform(NU_BOUNDS[0], NU_BOUNDS[1], 256),
            ]
        )
        mu, std = gp.predict(candidates, return_std=True)
        log_gamma, nu = candidates[int(np.argmax(mu + kappa * std))]
        observations_x.append([float(log_gamma), float(nu)])
        score, eval_duration = evaluate(log_gamma, nu)
        observations_y.append(score)
        append_history(
            history,
            start_time,
            len(observations_y),
            10**log_gamma,
            nu,
            score,
            trial_number=len(observations_y) - 1,
            eval_duration_sec=eval_duration,
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
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
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
            eval_start = time.perf_counter()
            last = validation_score(
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
            )
            eval_duration = time.perf_counter() - eval_start
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
                eval_duration_sec=eval_duration,
            )
            if trial.should_prune():
                history[-1]["status"] = "pruned"
                raise optuna.TrialPruned()
        return last

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, catch=(ValueError,))
    return dict(study.best_params), history


def summarize_metrics(results):
    summary = (
        results.groupby("method")[["Recall", "F-1", "AUC"]]
        .agg(["mean", "sem"])
        .reindex(METHODS)
    )
    return summary


def write_latex_rows(summary, path):
    lines = []
    for metric in ["Recall", "F-1", "AUC"]:
        row_values = []
        means = [summary.loc[m, (metric, "mean")] for m in METHODS]
        best_idx = int(np.argmax(means))
        for i, method in enumerate(METHODS):
            mean = summary.loc[method, (metric, "mean")]
            sem = summary.loc[method, (metric, "sem")]
            value = f"{mean:.4f} ({sem:.4f})"
            if i == best_idx:
                value = f"\\textbf{{{value}}}"
            row_values.append(value)
        lines.append(f"{metric} & " + " & ".join(row_values) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


