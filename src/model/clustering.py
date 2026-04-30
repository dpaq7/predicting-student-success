"""Student archetype clustering utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline

from src.model.evaluation import RANDOM_STATE, build_preprocessor


def evaluate_kmeans_clusters(
    data: pd.DataFrame,
    columns: list[str],
    *,
    cluster_counts: range = range(3, 7),
) -> tuple[pd.DataFrame, Pipeline]:
    """Evaluate k-means cluster counts and return the best fitted pipeline."""
    x = data[columns]
    rows: list[dict[str, float | int]] = []
    fitted: dict[int, Pipeline] = {}

    for k in cluster_counts:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(data, columns, scale_numeric=True)),
                (
                    "cluster",
                    KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE),
                ),
            ]
        )
        labels = pipeline.fit_predict(x)
        transformed = pipeline.named_steps["preprocess"].transform(x)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(transformed, labels),
                "davies_bouldin": davies_bouldin_score(transformed, labels),
                "inertia": pipeline.named_steps["cluster"].inertia_,
            }
        )
        fitted[k] = pipeline

    metrics = pd.DataFrame(rows)
    best_k = int(
        metrics.sort_values(
            ["silhouette", "davies_bouldin"], ascending=[False, True]
        ).iloc[0]["k"]
    )
    return metrics, fitted[best_k]


def cluster_profile_table(
    data: pd.DataFrame,
    columns: list[str],
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Summarize cluster-level student archetype characteristics."""
    profiled = data.copy()
    profiled["cluster"] = pipeline.predict(data[columns])
    summary_columns = [
        "cluster",
        "student_count",
        "pass_rate",
        "mean_G3",
        "mean_failures",
        "mean_absences",
        "mean_studytime",
        "mean_goout",
        "mean_Dalc",
        "mean_Walc",
    ]
    rows: list[dict[str, float | int]] = []
    for cluster_id, group in profiled.groupby("cluster"):
        rows.append(
            {
                "cluster": int(cluster_id),
                "student_count": int(len(group)),
                "pass_rate": float(group["pass_fail"].mean()),
                "mean_G3": float(group["G3"].mean()),
                "mean_failures": float(group["failures"].mean()),
                "mean_absences": float(group["absences"].mean()),
                "mean_studytime": float(group["studytime"].mean()),
                "mean_goout": float(group["goout"].mean()),
                "mean_Dalc": float(group["Dalc"].mean()),
                "mean_Walc": float(group["Walc"].mean()),
            }
        )
    return pd.DataFrame(rows)[summary_columns].sort_values("cluster")

