#!/usr/bin/env python
# coding: utf-8

"""Summarize final overlap experiments for Grid, Random, and BO only."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["grid", "random", "bayesian"]
METHOD_LABELS = {
    "grid": "Grid Search",
    "random": "Random Search",
    "bayesian": "BO",
}
COLORS = {
    "grid": "#4c78a8",
    "random": "#f58518",
    "bayesian": "#54a24b",
}
MARKERS = {
    "grid": "o",
    "random": "s",
    "bayesian": "^",
}
METRIC_LABELS = {
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "AUC",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "Review_Overlap_RatioSep",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def method_label(method):
    return METHOD_LABELS.get(method, method)


def ratio_label(outlier_fraction):
    normal = int(round((1.0 - outlier_fraction) * 100))
    anomaly = int(round(outlier_fraction * 100))
    return f"{normal}:{anomaly}"


def add_ordering_columns(df):
    out = df.copy()
    out["method_label"] = out["method"].map(METHOD_LABELS)
    out["ratio_label"] = out["outlier_fraction"].map(ratio_label)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["outlier_fraction", "separation", "method"])


def mean_sem_text(mean, sem, digits=3):
    return f"{mean:.{digits}f} ({sem:.{digits}f})"


def pivot_metric_table(summary, metric):
    metric_df = summary[summary["metric"] == metric].copy()
    rows = []
    for (outlier_fraction, separation), group in metric_df.groupby(["outlier_fraction", "separation"]):
        row = {
            "normal_to_anomaly_ratio": ratio_label(outlier_fraction),
            "outlier_fraction": outlier_fraction,
            "separation": separation,
        }
        best = group.loc[group["mean"].idxmax()]
        for method in METHOD_ORDER:
            method_row = group[group["method"] == method].iloc[0]
            row[METHOD_LABELS[method]] = mean_sem_text(method_row["mean"], method_row["sem"])
        row["winner"] = METHOD_LABELS[best["method"]]
        row["winner_mean"] = best["mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["outlier_fraction", "separation"], ascending=[True, False])


def pivot_value_table(df, value_col, sem_col, value_name, digits=3):
    rows = []
    for (outlier_fraction, separation), group in df.groupby(["outlier_fraction", "separation"]):
        row = {
            "normal_to_anomaly_ratio": ratio_label(outlier_fraction),
            "outlier_fraction": outlier_fraction,
            "separation": separation,
        }
        best_idx = group[value_col].idxmin() if "time" in value_col or "trials" in value_col else group[value_col].idxmax()
        best = group.loc[best_idx]
        for method in METHOD_ORDER:
            method_row = group[group["method"] == method].iloc[0]
            row[METHOD_LABELS[method]] = mean_sem_text(method_row[value_col], method_row[sem_col], digits=digits)
        row["best_method"] = METHOD_LABELS[best["method"]]
        row[f"best_{value_name}"] = best[value_col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["outlier_fraction", "separation"], ascending=[True, False])


def winner_tables(summary):
    rows = []
    for metric in ["recall", "f1", "roc_auc"]:
        metric_df = summary[summary["metric"] == metric].copy()
        for (outlier_fraction, separation), group in metric_df.groupby(["outlier_fraction", "separation"]):
            best = group.loc[group["mean"].idxmax()]
            rows.append({
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "normal_to_anomaly_ratio": ratio_label(outlier_fraction),
                "outlier_fraction": outlier_fraction,
                "separation": separation,
                "winner": METHOD_LABELS[best["method"]],
                "winner_mean": best["mean"],
                "winner_sem": best["sem"],
            })
    winners = pd.DataFrame(rows).sort_values(["metric", "outlier_fraction", "separation"], ascending=[True, True, False])
    counts = (
        winners.groupby(["metric", "metric_label", "winner"])
        .size()
        .reset_index(name="win_count")
        .sort_values(["metric", "win_count"], ascending=[True, False])
    )
    return winners, counts


def overall_performance(summary):
    rows = []
    for (metric, method), group in summary.groupby(["metric", "method"], observed=True):
        rows.append({
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "mean_over_scenarios": group["mean"].mean(),
            "std_over_scenarios": group["mean"].std(ddof=1),
            "min_scenario_mean": group["mean"].min(),
            "max_scenario_mean": group["mean"].max(),
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["metric", "method"])


def overall_time(summary):
    time_df = summary[summary["metric"] == "f1"].copy()
    rows = []
    for method, group in time_df.groupby("method", observed=True):
        rows.append({
            "method": method,
            "method_label": METHOD_LABELS[method],
            "mean_search_time_sec_per_fold": group["time_mean_sec"].mean(),
            "std_search_time_sec_per_fold": group["time_mean_sec"].std(ddof=1),
            "budget_evaluations": group["budget_evaluations"].iloc[0],
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values("method")


def overall_budget(budget):
    cols = [
        "trials_to_best_mean",
        "evals_to_best_mean",
        "time_to_best_sec_mean",
        "best_score_mean",
    ]
    rows = []
    for method, group in budget.groupby("method", observed=True):
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
        }
        for col in cols:
            row[f"{col}_over_scenarios"] = group[col].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values("method")


def ratio_ticks(df):
    ratios = sorted(df["outlier_fraction"].unique())
    labels = [ratio_label(ratio) for ratio in ratios]
    return ratios, labels


def separations(df):
    return sorted(df["separation"].unique(), reverse=True)


def plot_metric(summary, metric, output_path):
    metric_df = summary[summary["metric"] == metric].copy()
    ratios, labels = ratio_ticks(metric_df)
    sep_values = separations(metric_df)

    fig, axes = plt.subplots(1, len(sep_values), figsize=(14.2, 4.5), sharey=True)
    if len(sep_values) == 1:
        axes = [axes]

    for ax, separation in zip(axes, sep_values):
        sep_df = metric_df[metric_df["separation"] == separation]
        for method in METHOD_ORDER:
            method_df = sep_df[sep_df["method"] == method].sort_values("outlier_fraction")
            ax.errorbar(
                method_df["outlier_fraction"],
                method_df["mean"],
                yerr=method_df["sem"],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.8,
                markersize=6,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"Separation = {separation:g}")
        ax.set_xlabel("Normal:Anomaly ratio")
        ax.set_xticks(ratios)
        ax.set_xticklabels(labels, rotation=25)
        ax.grid(True, linewidth=0.45, alpha=0.35)

    axes[0].set_ylabel(METRIC_LABELS[metric])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{METRIC_LABELS[metric]}: Grid Search vs Random Search vs BO", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_value_by_ratio(df, value_col, sem_col, ylabel, title, output_path):
    ratios, labels = ratio_ticks(df)
    sep_values = separations(df)

    fig, axes = plt.subplots(1, len(sep_values), figsize=(14.2, 4.5), sharey=True)
    if len(sep_values) == 1:
        axes = [axes]

    for ax, separation in zip(axes, sep_values):
        sep_df = df[df["separation"] == separation]
        for method in METHOD_ORDER:
            method_df = sep_df[sep_df["method"] == method].sort_values("outlier_fraction")
            ax.errorbar(
                method_df["outlier_fraction"],
                method_df[value_col],
                yerr=method_df[sem_col],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=1.8,
                markersize=6,
                capsize=3,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"Separation = {separation:g}")
        ax.set_xlabel("Normal:Anomaly ratio")
        ax.set_xticks(ratios)
        ax.set_xticklabels(labels, rotation=25)
        ax.grid(True, linewidth=0.45, alpha=0.35)

    axes[0].set_ylabel(ylabel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_overall_performance(overall, output_path):
    pivot = overall.pivot(index="metric_label", columns="method", values="mean_over_scenarios")
    metric_order = ["F1-score", "AUC"]
    x = range(len(metric_order))
    width = 0.23

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    offsets = [-width, 0, width]
    for offset, method in zip(offsets, METHOD_ORDER):
        values = [pivot.loc[metric, method] for metric in metric_order]
        bars = ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            alpha=0.92,
        )
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=8, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_order)
    ax.set_ylabel("Mean over 15 scenarios")
    ax.set_ylim(0, 1.02)
    ax.set_title("Overall Performance")
    ax.grid(axis="y", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def plot_overall_time_budget(time_overall, budget_overall, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))

    time_values = time_overall.set_index("method").loc[METHOD_ORDER]
    bars = axes[0].bar(
        time_values["method_label"],
        time_values["mean_search_time_sec_per_fold"],
        color=[COLORS[m] for m in METHOD_ORDER],
    )
    axes[0].bar_label(bars, labels=[f"{v:.2f}" for v in time_values["mean_search_time_sec_per_fold"]], padding=3)
    axes[0].set_ylabel("Seconds per fold")
    axes[0].set_title("Search Time")
    axes[0].grid(axis="y", linewidth=0.45, alpha=0.35)

    budget_values = budget_overall.set_index("method").loc[METHOD_ORDER]
    bars = axes[1].bar(
        budget_values["method_label"],
        budget_values["trials_to_best_mean_over_scenarios"],
        color=[COLORS[m] for m in METHOD_ORDER],
    )
    axes[1].bar_label(bars, labels=[f"{v:.1f}" for v in budget_values["trials_to_best_mean_over_scenarios"]], padding=3)
    axes[1].set_ylabel("Trials")
    axes[1].set_title("Budget to First Best Score")
    axes[1].grid(axis="y", linewidth=0.45, alpha=0.35)

    fig.suptitle("Overall Search Cost", y=0.98, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def markdown_table(df, columns=None):
    out = df if columns is None else df[columns]
    out = out.copy()
    headers = list(out.columns)
    rows = []
    for _, row in out.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)

    def clean(value):
        return value.replace("|", "\\|").replace("\n", " ")

    lines = []
    lines.append("| " + " | ".join(clean(str(header)) for header in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(
    output_path,
    f1_table,
    auc_table,
    recall_table,
    winners,
    win_counts,
    perf_overall,
    time_overall,
    budget_overall,
):
    def metric_overall(metric):
        rows = perf_overall[perf_overall["metric"] == metric].copy()
        return rows[["method_label", "mean_over_scenarios", "std_over_scenarios", "min_scenario_mean", "max_scenario_mean"]]

    lines = []
    lines.append("# Grid Search / Random Search / BO 비교 요약")
    lines.append("")
    lines.append("## 기준")
    lines.append("")
    lines.append("- 대상 실험: `Review_Overlap_RatioSep` 최종 overlap 합성 데이터 실험.")
    lines.append("- 비교 방법: Grid Search, Random Search, Bayesian Optimization(BO).")
    lines.append("- 제외 방법: Hyperband, BOHB.")
    lines.append("- 정상:이상 비율: 95:5, 90:10, 80:20, 70:30, 60:40.")
    lines.append("- separation: 2.0, 1.5, 1.0. separation이 작을수록 경계가 더 모호함.")
    lines.append("- 반복: 100회, 내부 검증: 5-fold.")
    lines.append("")

    lines.append("## 전체 평균 성능")
    lines.append("")
    overall_short = perf_overall[["metric_label", "method_label", "mean_over_scenarios", "std_over_scenarios"]].copy()
    lines.append(markdown_table(overall_short))
    lines.append("")

    lines.append("## F1 시나리오별 결과")
    lines.append("")
    lines.append(markdown_table(f1_table[["normal_to_anomaly_ratio", "separation", "Grid Search", "Random Search", "BO", "winner"]]))
    lines.append("")

    lines.append("## AUC 시나리오별 결과")
    lines.append("")
    lines.append(markdown_table(auc_table[["normal_to_anomaly_ratio", "separation", "Grid Search", "Random Search", "BO", "winner"]]))
    lines.append("")

    lines.append("## Recall 시나리오별 결과")
    lines.append("")
    lines.append(markdown_table(recall_table[["normal_to_anomaly_ratio", "separation", "Grid Search", "Random Search", "BO", "winner"]]))
    lines.append("")

    lines.append("## 우승 횟수")
    lines.append("")
    lines.append(markdown_table(win_counts))
    lines.append("")

    lines.append("## 탐색 시간")
    lines.append("")
    lines.append(markdown_table(time_overall[["method_label", "mean_search_time_sec_per_fold", "std_search_time_sec_per_fold", "budget_evaluations"]]))
    lines.append("")

    lines.append("## 최적 검증 점수 도달 budget")
    lines.append("")
    budget_short = budget_overall[[
        "method_label",
        "trials_to_best_mean_over_scenarios",
        "evals_to_best_mean_over_scenarios",
        "time_to_best_sec_mean_over_scenarios",
        "best_score_mean_over_scenarios",
    ]]
    lines.append(markdown_table(budget_short))
    lines.append("")

    f1_wins = winners[winners["metric"] == "f1"]["winner"].value_counts().to_dict()
    auc_wins = winners[winners["metric"] == "roc_auc"]["winner"].value_counts().to_dict()
    lines.append("## 해석")
    lines.append("")
    lines.append(f"- F1 기준 우승 횟수: {f1_wins}.")
    lines.append(f"- AUC 기준 우승 횟수: {auc_wins}.")
    lines.append("- 세 방법만 비교하면, 최종 테스트 성능에서는 Random Search와 BO가 조건에 따라 번갈아 우세하다.")
    lines.append("- 쉬운 조건(separation=2.0)에서는 BO가 F1에서 강한 편이고, 경계가 더 모호한 조건에서는 Random Search가 자주 우세하다.")
    lines.append("- 탐색 시간과 최적 검증 점수 도달 budget은 Random Search가 가장 작다.")
    lines.append("- 따라서 review 대응 문장에서는 BO가 항상 최고라고 쓰기보다, Grid/Random 대비 BO의 장점과 overlap 조건에서의 한계를 함께 제시하는 것이 안전하다.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    args = parse_args()
    root = args.root
    output_dir = args.output_dir or root / "Grid_Random_BO"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(root / "overlap_summary_with_time.csv")
    budget = pd.read_csv(root / "budget_to_best_summary.csv")
    summary = add_ordering_columns(summary[summary["method"].isin(METHOD_ORDER)])
    budget = add_ordering_columns(budget[budget["method"].isin(METHOD_ORDER)])

    summary.to_csv(output_dir / "performance_long_grid_random_bo.csv", index=False)
    budget.to_csv(output_dir / "budget_to_best_grid_random_bo.csv", index=False)

    recall_table = pivot_metric_table(summary, "recall")
    f1_table = pivot_metric_table(summary, "f1")
    auc_table = pivot_metric_table(summary, "roc_auc")
    recall_table.to_csv(output_dir / "recall_table_grid_random_bo.csv", index=False)
    f1_table.to_csv(output_dir / "f1_table_grid_random_bo.csv", index=False)
    auc_table.to_csv(output_dir / "auc_table_grid_random_bo.csv", index=False)

    time_df = summary[summary["metric"] == "f1"].copy()
    time_table = pivot_value_table(
        time_df,
        "time_mean_sec",
        "time_sem_sec",
        "search_time_sec",
        digits=2,
    )
    trial_budget_table = pivot_value_table(
        budget,
        "trials_to_best_mean",
        "trials_to_best_sem",
        "trials_to_best",
        digits=1,
    )
    time_to_best_table = pivot_value_table(
        budget,
        "time_to_best_sec_mean",
        "time_to_best_sec_sem",
        "time_to_best_sec",
        digits=2,
    )
    time_table.to_csv(output_dir / "search_time_table_grid_random_bo.csv", index=False)
    trial_budget_table.to_csv(output_dir / "budget_trials_table_grid_random_bo.csv", index=False)
    time_to_best_table.to_csv(output_dir / "budget_time_table_grid_random_bo.csv", index=False)

    winners, win_counts = winner_tables(summary)
    perf_overall = overall_performance(summary)
    time_overall = overall_time(summary)
    budget_overall = overall_budget(budget)
    winners.to_csv(output_dir / "winners_by_scenario_grid_random_bo.csv", index=False)
    win_counts.to_csv(output_dir / "win_counts_grid_random_bo.csv", index=False)
    perf_overall.to_csv(output_dir / "overall_performance_grid_random_bo.csv", index=False)
    time_overall.to_csv(output_dir / "overall_search_time_grid_random_bo.csv", index=False)
    budget_overall.to_csv(output_dir / "overall_budget_to_best_grid_random_bo.csv", index=False)

    for metric in ["f1", "roc_auc"]:
        plot_metric(summary, metric, output_dir / f"{metric}_grid_random_bo_by_ratio_separation.png")

    plot_value_by_ratio(
        time_df,
        "time_mean_sec",
        "time_sem_sec",
        "Mean search time per fold (sec)",
        "Search Time: Grid Search vs Random Search vs BO",
        output_dir / "search_time_grid_random_bo_by_ratio_separation.png",
    )
    plot_value_by_ratio(
        budget,
        "trials_to_best_mean",
        "trials_to_best_sem",
        "Trials to first best score",
        "Budget to First Best Score",
        output_dir / "budget_trials_grid_random_bo_by_ratio_separation.png",
    )
    plot_value_by_ratio(
        budget,
        "time_to_best_sec_mean",
        "time_to_best_sec_sem",
        "Seconds to first best score",
        "Time to First Best Score",
        output_dir / "budget_time_grid_random_bo_by_ratio_separation.png",
    )
    plot_overall_performance(perf_overall, output_dir / "overall_performance_grid_random_bo.png")
    plot_overall_time_budget(time_overall, budget_overall, output_dir / "overall_search_cost_grid_random_bo.png")

    write_markdown(
        output_dir / "grid_random_bo_summary_ko.md",
        f1_table,
        auc_table,
        recall_table,
        winners,
        win_counts,
        perf_overall,
        time_overall,
        budget_overall,
    )
    print(f"Saved summary directory: {output_dir}")


if __name__ == "__main__":
    main()
