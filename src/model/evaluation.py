"""Reusable model evaluation utilities for the capstone analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.student_performance import split_feature_types


RANDOM_STATE = 42


@dataclass(frozen=True)
class ClassificationResult:
    """Cross-validated classification metrics for one model and feature set."""

    feature_set: str
    model: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fit_seconds: float


@dataclass(frozen=True)
class RegressionResult:
    """Cross-validated regression metrics for one model and feature set."""

    feature_set: str
    model: str
    rmse: float
    mae: float
    r2: float
    fit_seconds: float


def build_preprocessor(
    data: pd.DataFrame,
    columns: list[str],
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a dense preprocessing transformer for mixed tabular data."""
    numeric_features, categorical_features = split_feature_types(data, columns)
    numeric_transformer: str | StandardScaler = (
        StandardScaler() if scale_numeric else "passthrough"
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )


def classification_models() -> dict[str, object]:
    """Return the classification algorithms used in the capstone comparison."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=80,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def regression_models() -> dict[str, object]:
    """Return the regression algorithms used in the capstone comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=80,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_classifiers(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    folds: int = 3,
) -> pd.DataFrame:
    """Evaluate classifiers across temporal feature sets."""
    rows: list[ClassificationResult] = []
    y = data["pass_fail"].to_numpy()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for feature_set_name, columns in feature_sets.items():
        x = data[columns]
        for model_name, estimator in classification_models().items():
            fold_metrics: list[dict[str, float]] = []
            start = time.perf_counter()
            for train_index, test_index in cv.split(x, y):
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", build_preprocessor(data, columns)),
                        ("model", clone(estimator)),
                    ]
                )
                pipeline.fit(x.iloc[train_index], y[train_index])
                predictions = pipeline.predict(x.iloc[test_index])
                probabilities = _classification_probabilities(
                    pipeline, x.iloc[test_index], predictions
                )
                fold_metrics.append(
                    {
                        "accuracy": accuracy_score(y[test_index], predictions),
                        "balanced_accuracy": balanced_accuracy_score(
                            y[test_index], predictions
                        ),
                        "precision": precision_score(
                            y[test_index], predictions, zero_division=0
                        ),
                        "recall": recall_score(
                            y[test_index], predictions, zero_division=0
                        ),
                        "f1": f1_score(y[test_index], predictions, zero_division=0),
                        "roc_auc": roc_auc_score(y[test_index], probabilities),
                    }
                )
            elapsed = time.perf_counter() - start
            metrics = pd.DataFrame(fold_metrics).mean()
            rows.append(
                ClassificationResult(
                    feature_set=feature_set_name,
                    model=model_name,
                    accuracy=float(metrics["accuracy"]),
                    balanced_accuracy=float(metrics["balanced_accuracy"]),
                    precision=float(metrics["precision"]),
                    recall=float(metrics["recall"]),
                    f1=float(metrics["f1"]),
                    roc_auc=float(metrics["roc_auc"]),
                    fit_seconds=elapsed,
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def evaluate_regressors(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    folds: int = 3,
) -> pd.DataFrame:
    """Evaluate regressors across temporal feature sets."""
    rows: list[RegressionResult] = []
    y = data["G3"].to_numpy()
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for feature_set_name, columns in feature_sets.items():
        x = data[columns]
        for model_name, estimator in regression_models().items():
            fold_metrics: list[dict[str, float]] = []
            start = time.perf_counter()
            for train_index, test_index in cv.split(x):
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", build_preprocessor(data, columns)),
                        ("model", clone(estimator)),
                    ]
                )
                pipeline.fit(x.iloc[train_index], y[train_index])
                predictions = pipeline.predict(x.iloc[test_index])
                fold_metrics.append(
                    {
                        "rmse": float(
                            np.sqrt(mean_squared_error(y[test_index], predictions))
                        ),
                        "mae": mean_absolute_error(y[test_index], predictions),
                        "r2": r2_score(y[test_index], predictions),
                    }
                )
            elapsed = time.perf_counter() - start
            metrics = pd.DataFrame(fold_metrics).mean()
            rows.append(
                RegressionResult(
                    feature_set=feature_set_name,
                    model=model_name,
                    rmse=float(metrics["rmse"]),
                    mae=float(metrics["mae"]),
                    r2=float(metrics["r2"]),
                    fit_seconds=elapsed,
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def fit_pipeline(data: pd.DataFrame, columns: list[str], estimator: object) -> Pipeline:
    """Fit a preprocessing-plus-model pipeline on the full dataset."""
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(data, columns)),
            ("model", clone(estimator)),
        ]
    )
    pipeline.fit(data[columns], data["pass_fail"])
    return pipeline


def permutation_importance_table(
    pipeline: Pipeline,
    data: pd.DataFrame,
    columns: list[str],
    *,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Compute permutation importance for educator-facing explanations."""
    result = permutation_importance(
        pipeline,
        data[columns],
        data["pass_fail"],
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _classification_probabilities(
    pipeline: Pipeline,
    x: pd.DataFrame,
    predictions: np.ndarray,
) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return predictions.astype(float)
