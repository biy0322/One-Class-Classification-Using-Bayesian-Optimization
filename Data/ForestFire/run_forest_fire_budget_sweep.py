import argparse
import math
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.forest_fire_data import load_dataset, make_split, summarize_metrics
from common.ocsvm_hpo import (
    GAMMA_BOUNDS,
    NU_BOUNDS,
    append_history,
    bayes_search,
    fit_and_test,
    hyperband_search,
    random_search,
    summarize_history,
    validation_score,
)


METHODS = ["GC", "RC", "BO", "HB", "BOHB"]
TARGETS = [0.980, 0.990, 0.995]
optuna.logging.set_verbosity(optuna.logging.WARNING)


def grid_search_budget(
    x_fit,
    x_val,
    y_val,
    budget,
    validation_metric="f_beta",
    validation_beta=1.0,
    validation_average="binary",
):
    n_gamma = int(math.ceil(math.sqrt(budget)))
    n_nu = int(math.ceil(budget / n_gamma))
    gammas = np.logspace(math.log10(GAMMA_BOUNDS[0]), math.log10(GAMMA_BOUNDS[1]), n_gamma)
    nus = np.linspace(NU_BOUNDS[0], NU_BOUNDS[1], n_nu)
    configs = list((float(g), float(n)) for g in gammas for n in nus)[:budget]

    best = (-np.inf, None)
    history = []
    start_time = __import__("time").perf_counter()
    for idx, (gamma, nu) in enumerate(configs, start=1):
        eval_start = __import__("time").perf_counter()
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
        eval_duration = __import__("time").perf_counter() - eval_start
        if score > best[0]:
            best = (score, {"gamma": gamma, "nu": nu})
        append_history(
            history,
            start_time,
            idx,
            gamma,
            nu,
            score,
            trial_number=idx - 1,
            eval_duration_sec=eval_duration,
        )
    return best[1], history


def mean_sem(values):
    series = pd.Series(values, dtype=float)
    return series.mean(), series.sem(ddof=1)


def fmt_mean_sem(mean, sem, digits=3):
    if pd.isna(mean):
        return "Not reached"
    if pd.isna(sem):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def summarize_by_budget(results, time_to_best):
    rows = []
    for budget in sorted(results["budget"].unique()):
        for method in METHODS:
            perf_group = results[(results["budget"] == budget) & (results["method"] == method)]
            time_group = time_to_best[(time_to_best["budget"] == budget) & (time_to_best["method"] == method)]
            row = {"budget": int(budget), "method": method}
            for col in ["Recall", "F-1", "AUC"]:
                mean, sem = mean_sem(perf_group[col])
                row[f"{col}_mean"] = mean
                row[f"{col}_sem"] = sem
            for col in ["best_validation_score", "configs_to_best", "full_validation_evals", "total_observed_evals"]:
                mean, sem = mean_sem(time_group[col])
                row[f"{col}_mean"] = mean
                row[f"{col}_sem"] = sem
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_relative_targets(history, budget):
    history = history[(history["budget"] == budget) & (history["full_eval"].astype(bool))]
    per_repeat = []
    for repeat, rep_group in history.groupby("repeat", sort=False):
        split_best = rep_group["score"].max()
        for target in TARGETS:
            target_score = target * split_best
            for method in METHODS:
                group = rep_group[rep_group["method"] == method].sort_values("eval_index")
                reached = group[group["best_so_far"] >= target_score]
                if reached.empty:
                    per_repeat.append(
                        {
                            "budget": budget,
                            "repeat": repeat,
                            "target_fraction": target,
                            "method": method,
                            "target_validation_f1": target_score,
                            "reached": False,
                            "configs_to_target": np.nan,
                        }
                    )
                    continue
                first = reached.iloc[0]
                trial_number = first.get("trial_number")
                configs_to_target = first["eval_index"] if pd.isna(trial_number) else int(trial_number) + 1
                per_repeat.append(
                    {
                        "budget": budget,
                        "repeat": repeat,
                        "target_fraction": target,
                        "method": method,
                        "target_validation_f1": target_score,
                        "reached": True,
                        "configs_to_target": configs_to_target,
                    }
                )

    per_repeat = pd.DataFrame(per_repeat)
    rows = []
    for target in TARGETS:
        for method in METHODS:
            group = per_repeat[(per_repeat["target_fraction"] == target) & (per_repeat["method"] == method)]
            reached = group[group["reached"]]
            target_mean, target_sem = mean_sem(group["target_validation_f1"])
            config_mean, config_sem = mean_sem(reached["configs_to_target"])
            rows.append(
                {
                    "target_fraction": target,
                    "method": method,
                    "n_repeats": int(len(group)),
                    "n_reached": int(group["reached"].sum()),
                    "reach_rate": float(group["reached"].mean()) if len(group) else np.nan,
                    "target_validation_f1_mean": target_mean,
                    "target_validation_f1_sem": target_sem,
                    "configs_to_target_mean": config_mean,
                    "configs_to_target_sem": config_sem,
                }
            )
    return per_repeat, pd.DataFrame(rows)


def write_budget_latex(summary, path):
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{int(row['budget'])} & {row['method']} & "
            f"{fmt_mean_sem(row['F-1_mean'], row['F-1_sem'], 3)} & "
            f"{fmt_mean_sem(row['AUC_mean'], row['AUC_sem'], 3)} & "
            f"{fmt_mean_sem(row['best_validation_score_mean'], row['best_validation_score_sem'], 3)} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_target_latex(summary, path):
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row['target_fraction']:.3f} & {row['method']} & "
            f"{int(row['n_reached'])}/{int(row['n_repeats'])} & "
            f"{fmt_mean_sem(row['configs_to_target_mean'], row['configs_to_target_sem'], 1)} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(summary, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for method in METHODS:
        group = summary[summary["method"] == method].sort_values("budget")
        axes[0].errorbar(group["budget"], group["F-1_mean"], yerr=group["F-1_sem"], marker="o", label=method)
        axes[1].errorbar(group["budget"], group["AUC_mean"], yerr=group["AUC_sem"], marker="o", label=method)
    axes[0].set_ylabel("Final test F1")
    axes[1].set_ylabel("Final test AUC")
    for ax in axes:
        ax.set_xlabel("Configuration budget")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("ForestFire/results_budget_sweep"))
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--validation-metric", choices=["f1", "f_beta"], default="f_beta")
    parser.add_argument("--validation-beta", type=float, default=1.0)
    parser.add_argument("--validation-average", choices=["binary", "macro"], default="binary")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_raw, y, data_file = load_dataset("forest_fire", REPO_ROOT / "ForestFire")

    rows = []
    history_rows = []
    time_rows = []

    for budget in args.budgets:
        print(f"\nBudget {budget}", flush=True)
        for repeat in range(args.n_repeats):
            seed = args.seed + repeat
            split = make_split(x_raw, y, seed, args.test_size, args.val_size)
            rng = np.random.default_rng(seed + budget * 100)
            searches = {
                "GC": lambda: grid_search_budget(
                    split["x_search_fit"],
                    split["x_val"],
                    split["y_val"],
                    budget,
                    validation_metric=args.validation_metric,
                    validation_beta=args.validation_beta,
                    validation_average=args.validation_average,
                ),
                "RC": lambda: random_search(
                    split["x_search_fit"],
                    split["x_val"],
                    split["y_val"],
                    rng,
                    budget,
                    validation_metric=args.validation_metric,
                    validation_beta=args.validation_beta,
                    validation_average=args.validation_average,
                ),
                "BO": lambda: bayes_search(
                    split["x_search_fit"],
                    split["x_val"],
                    split["y_val"],
                    rng,
                    budget,
                    validation_metric=args.validation_metric,
                    validation_beta=args.validation_beta,
                    validation_average=args.validation_average,
                ),
                "HB": lambda: hyperband_search(
                    split["x_search_fit"],
                    split["x_val"],
                    split["y_val"],
                    seed,
                    budget,
                    "random",
                    validation_metric=args.validation_metric,
                    validation_beta=args.validation_beta,
                    validation_average=args.validation_average,
                ),
                "BOHB": lambda: hyperband_search(
                    split["x_search_fit"],
                    split["x_val"],
                    split["y_val"],
                    seed,
                    budget,
                    "tpe",
                    validation_metric=args.validation_metric,
                    validation_beta=args.validation_beta,
                    validation_average=args.validation_average,
                ),
            }

            print(f"  repeat {repeat + 1}/{args.n_repeats}", flush=True)
            for method in METHODS:
                params, history = searches[method]()
                metrics = fit_and_test(
                    split["x_final_fit"],
                    split["x_test"],
                    split["y_test"],
                    **params,
                )
                time_summary = summarize_history(history)
                rows.append(
                    {
                        "budget": budget,
                        "repeat": repeat,
                        "method": method,
                        "gamma": params["gamma"],
                        "nu": params["nu"],
                        **metrics,
                    }
                )
                time_rows.append(
                    {
                        "budget": budget,
                        "repeat": repeat,
                        "method": method,
                        **time_summary,
                    }
                )
                for h in history:
                    history_rows.append(
                        {
                            "budget": budget,
                            "repeat": repeat,
                            "method": method,
                            **h,
                        }
                    )
                print(
                    f"    {method}: F1={metrics['F-1']:.4f}, AUC={metrics['AUC']:.4f}, "
                    f"best_val={time_summary['best_validation_score']:.4f}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    history = pd.DataFrame(history_rows)
    time_to_best = pd.DataFrame(time_rows)
    budget_summary = summarize_by_budget(results, time_to_best)
    target_per_repeat, target_summary = summarize_relative_targets(history, budget=max(args.budgets))

    results.to_csv(args.output_dir / "forest_fire_budget_sweep_results.csv", index=False)
    history.to_csv(args.output_dir / "forest_fire_budget_sweep_history.csv", index=False)
    time_to_best.to_csv(args.output_dir / "forest_fire_budget_sweep_time_to_best.csv", index=False)
    budget_summary.to_csv(args.output_dir / "forest_fire_budget_sweep_summary.csv", index=False)
    target_per_repeat.to_csv(args.output_dir / "forest_fire_relative_target_per_repeat.csv", index=False)
    target_summary.to_csv(args.output_dir / "forest_fire_relative_target_summary.csv", index=False)
    write_budget_latex(budget_summary, args.output_dir / "forest_fire_budget_sweep_latex_rows.txt")
    write_target_latex(target_summary, args.output_dir / "forest_fire_relative_target_latex_rows.txt")
    make_plot(budget_summary, args.output_dir / "forest_fire_budget_sweep_f1_auc.png")

    print("\nBudget summary")
    print(budget_summary.to_string(index=False))
    print("\nRelative target summary")
    print(target_summary.to_string(index=False))


if __name__ == "__main__":
    main()
