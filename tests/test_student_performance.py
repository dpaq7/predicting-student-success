"""Tests for the student performance capstone pipeline."""

from pathlib import Path

from src.data.student_performance import (
    RAW_FILENAMES,
    T0_COLUMNS,
    T1_COLUMNS,
    T2_COLUMNS,
    load_student_performance_data,
)
from src.model.evaluation import classification_models, regression_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"


def test_raw_data_files_are_available() -> None:
    """The reproducible analysis depends on both UCI CSV files."""
    for filename in RAW_FILENAMES.values():
        assert (RAW_DATA_DIR / filename).exists()


def test_combined_dataset_shape_and_targets() -> None:
    """Combined data should match the known UCI dataset structure."""
    data = load_student_performance_data(RAW_DATA_DIR)

    assert data.shape[0] == 1_044
    assert data["subject"].nunique() == 2
    assert data["pass_fail"].isin([0, 1]).all()
    assert data["pass_fail"].mean() > 0.70


def test_temporal_feature_sets_do_not_leak_future_grades() -> None:
    """Earlier feature sets must not include later current-year grades."""
    assert "G1" not in T0_COLUMNS
    assert "G2" not in T0_COLUMNS
    assert "G3" not in T0_COLUMNS
    assert "G1" in T1_COLUMNS
    assert "G2" not in T1_COLUMNS
    assert "G3" not in T1_COLUMNS
    assert "G1" in T2_COLUMNS
    assert "G2" in T2_COLUMNS
    assert "G3" not in T2_COLUMNS


def test_model_suites_are_defined() -> None:
    """The capstone uses baseline, benchmark, and modern sklearn models."""
    assert set(classification_models()) == {
        "Logistic Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    }
    assert set(regression_models()) == {
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Hist Gradient Boosting",
    }
