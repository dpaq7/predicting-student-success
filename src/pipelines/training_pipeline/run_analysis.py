"""Generate capstone analysis outputs from the UCI student performance data."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.student_performance import (
    T0_COLUMNS,
    TEMPORAL_FEATURE_SETS,
    load_student_performance_data,
)
from src.model.clustering import cluster_profile_table, evaluate_kmeans_clusters
from src.model.evaluation import (
    classification_models,
    evaluate_classifiers,
    evaluate_regressors,
    fit_pipeline,
    permutation_importance_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"
PRIMARY_DATA_DIR = PROJECT_ROOT / "data" / "03_primary"
REPORTING_DIR = PROJECT_ROOT / "data" / "08_reporting"
FIGURES_DIR = REPORTING_DIR / "figures"
TABLES_DIR = REPORTING_DIR / "tables"


def main() -> None:
    """Run the full reproducible analysis and write report-ready artifacts."""
    _create_output_dirs()
    sns.set_theme(style="whitegrid", context="notebook")

    data = load_student_performance_data(RAW_DATA_DIR)
    data.to_csv(PRIMARY_DATA_DIR / "student_performance_combined.csv", index=False)

    data_summary = _write_data_summary(data)
    _write_eda_figures(data)

    classification = evaluate_classifiers(data, TEMPORAL_FEATURE_SETS)
    regression = evaluate_regressors(data, TEMPORAL_FEATURE_SETS)
    classification.to_csv(TABLES_DIR / "classification_metrics.csv", index=False)
    regression.to_csv(TABLES_DIR / "regression_metrics.csv", index=False)
    _write_metric_figures(classification, regression)

    best_row = classification.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
    best_model_name = str(best_row["model"])
    best_feature_set = str(best_row["feature_set"])
    best_columns = TEMPORAL_FEATURE_SETS[best_feature_set]
    best_estimator = classification_models()[best_model_name]

    pipeline = fit_pipeline(data, best_columns, best_estimator)
    importance = permutation_importance_table(pipeline, data, best_columns)
    importance.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)
    _write_importance_figure(importance)

    cluster_metrics, cluster_pipeline = evaluate_kmeans_clusters(data, T0_COLUMNS)
    cluster_profiles = cluster_profile_table(data, T0_COLUMNS, cluster_pipeline)
    cluster_metrics.to_csv(TABLES_DIR / "cluster_metrics.csv", index=False)
    cluster_profiles.to_csv(TABLES_DIR / "cluster_profiles.csv", index=False)
    _write_cluster_figures(cluster_metrics, cluster_profiles)

    _write_markdown_summary(
        data_summary=data_summary,
        classification=classification,
        regression=regression,
        importance=importance,
        cluster_metrics=cluster_metrics,
        cluster_profiles=cluster_profiles,
        best_model_name=best_model_name,
        best_feature_set=best_feature_set,
    )


def _create_output_dirs() -> None:
    for directory in [PRIMARY_DATA_DIR, REPORTING_DIR, FIGURES_DIR, TABLES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _write_data_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("subject")
        .agg(
            students=("G3", "size"),
            pass_rate=("pass_fail", "mean"),
            mean_G1=("G1", "mean"),
            mean_G2=("G2", "mean"),
            mean_G3=("G3", "mean"),
            median_G3=("G3", "median"),
            mean_absences=("absences", "mean"),
            mean_failures=("failures", "mean"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "subject": "combined",
                "students": len(data),
                "pass_rate": data["pass_fail"].mean(),
                "mean_G1": data["G1"].mean(),
                "mean_G2": data["G2"].mean(),
                "mean_G3": data["G3"].mean(),
                "median_G3": data["G3"].median(),
                "mean_absences": data["absences"].mean(),
                "mean_failures": data["failures"].mean(),
            }
        ]
    )
    output = pd.concat([summary, overall], ignore_index=True)
    output.to_csv(TABLES_DIR / "data_summary.csv", index=False)
    return output


def _write_eda_figures(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=data, x="G3", hue="subject", multiple="dodge", bins=21, ax=ax)
    ax.set_title("Final Grade Distribution by Subject")
    ax.set_xlabel("Final grade (G3)")
    ax.set_ylabel("Student count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grade_distribution_by_subject.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pass_rate = data.groupby("subject", as_index=False)["pass_fail"].mean()
    sns.barplot(data=pass_rate, x="subject", y="pass_fail", ax=ax)
    ax.axhline(data["pass_fail"].mean(), color="black", linestyle="--", linewidth=1)
    ax.set_title("Pass Rate by Subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Pass rate (G3 >= 10)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pass_rate_by_subject.png", dpi=160)
    plt.close(fig)

    numeric_columns = [
        "age",
        "Medu",
        "Fedu",
        "traveltime",
        "studytime",
        "failures",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        "G1",
        "G2",
        "G3",
    ]
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(data[numeric_columns].corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Heatmap for Numeric Features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="failures", y="G3", hue="subject", ax=ax)
    ax.set_title("Final Grade by Prior Failures")
    ax.set_xlabel("Number of prior failures")
    ax.set_ylabel("Final grade (G3)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grade_by_prior_failures.png", dpi=160)
    plt.close(fig)


def _write_metric_figures(classification: pd.DataFrame, regression: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=classification,
        x="feature_set",
        y="accuracy",
        hue="model",
        ax=ax,
    )
    ax.axhline(0.93, color="black", linestyle="--", linewidth=1, label="Cortez RF 93%")
    ax.axhline(0.80, color="gray", linestyle=":", linewidth=1, label="80% action threshold")
    ax.set_title("Classification Accuracy by Temporal Feature Set")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_accuracy_temporal.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=regression, x="feature_set", y="rmse", hue="model", ax=ax)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1, label="Cortez RMSE ~2.0")
    ax.set_title("Regression RMSE by Temporal Feature Set")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("RMSE on 0-20 grade scale")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_rmse_temporal.png", dpi=160)
    plt.close(fig)


def _write_importance_figure(importance: pd.DataFrame) -> None:
    top_features = importance.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_features["feature"], top_features["importance_mean"])
    ax.set_title("Top Permutation Importance Features")
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_permutation_importance.png", dpi=160)
    plt.close(fig)


def _write_cluster_figures(
    cluster_metrics: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=cluster_metrics, x="k", y="silhouette", marker="o", ax=ax)
    ax.set_title("K-Means Silhouette Score by Cluster Count")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_silhouette_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=cluster_profiles, x="cluster", y="pass_rate", ax=ax)
    ax.set_title("Pass Rate by Student Archetype Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_pass_rates.png", dpi=160)
    plt.close(fig)


def _write_markdown_summary(
    *,
    data_summary: pd.DataFrame,
    classification: pd.DataFrame,
    regression: pd.DataFrame,
    importance: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    best_model_name: str,
    best_feature_set: str,
) -> None:
    best_classification = classification.sort_values(
        ["roc_auc", "f1"], ascending=False
    ).iloc[0]
    best_regression = regression.sort_values("rmse").iloc[0]
    t0_best = (
        classification[classification["feature_set"] == "T0_no_current_grades"]
        .sort_values(["roc_auc", "f1"], ascending=False)
        .iloc[0]
    )
    first_actionable = (
        classification[classification["accuracy"] >= 0.80]
        .sort_values(["feature_set", "accuracy"], ascending=[True, False])
        .head(1)
    )
    timing_text = (
        "No temporal feature set reached 80% accuracy."
        if first_actionable.empty
        else (
            f"The first feature set to exceed 80% accuracy was "
            f"{first_actionable.iloc[0]['feature_set']} with "
            f"{first_actionable.iloc[0]['model']} "
            f"({first_actionable.iloc[0]['accuracy']:.3f} accuracy)."
        )
    )

    summary = f"""# Capstone Analysis Results Summary

Generated by `python -m src.pipelines.training_pipeline.run_analysis`.

## Dataset

{_markdown_table(data_summary.round(3))}

## RQ1: Algorithm Benchmarking

Best classification result: **{best_classification['model']}** on **{best_classification['feature_set']}** with accuracy **{best_classification['accuracy']:.3f}**, F1 **{best_classification['f1']:.3f}**, and ROC-AUC **{best_classification['roc_auc']:.3f}**.

Best regression result: **{best_regression['model']}** on **{best_regression['feature_set']}** with RMSE **{best_regression['rmse']:.3f}**, MAE **{best_regression['mae']:.3f}**, and R2 **{best_regression['r2']:.3f}**.

## RQ2: Early Warning Timing

Best no-current-grade model: **{t0_best['model']}** with accuracy **{t0_best['accuracy']:.3f}**, balanced accuracy **{t0_best['balanced_accuracy']:.3f}**, and ROC-AUC **{t0_best['roc_auc']:.3f}**.

{timing_text}

## RQ3: Educator-Facing Explainability

The selected explanation model is **{best_model_name}** using **{best_feature_set}**. The highest-impact permutation features were:

{_markdown_table(importance.head(10).round(4))}

Interpretation note: permutation importance is used here as a package-light, model-agnostic explainability method. If SHAP is installed later, SHAP summary plots should be added for richer local explanations.

## RQ4: Student Archetypes

Cluster validation:

{_markdown_table(cluster_metrics.round(3))}

Cluster profiles:

{_markdown_table(cluster_profiles.round(3))}

## Report-Ready Figures

- `figures/grade_distribution_by_subject.png`
- `figures/pass_rate_by_subject.png`
- `figures/numeric_correlation_heatmap.png`
- `figures/grade_by_prior_failures.png`
- `figures/classification_accuracy_temporal.png`
- `figures/regression_rmse_temporal.png`
- `figures/top_permutation_importance.png`
- `figures/cluster_silhouette_scores.png`
- `figures/cluster_pass_rates.png`
"""
    (REPORTING_DIR / "analysis_summary.md").write_text(summary)


def _markdown_table(data: pd.DataFrame) -> str:
    """Render a small pandas DataFrame as a GitHub-flavored Markdown table."""
    headers = [str(column) for column in data.columns]
    rows = [
        [str(value) for value in row]
        for row in data.astype(object).where(pd.notna(data), "").to_numpy()
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
