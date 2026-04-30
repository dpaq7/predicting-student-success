"""Data loading and feature definitions for the UCI student performance data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PASSING_GRADE = 10
RAW_FILENAMES = {
    "math": "student-mat.csv",
    "portuguese": "student-por.csv",
}

GRADE_COLUMNS = ["G1", "G2", "G3"]
TARGET_COLUMNS = ["G3", "pass_fail"]

T0_COLUMNS = [
    "subject",
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
]
T1_COLUMNS = [*T0_COLUMNS, "absences", "G1"]
T2_COLUMNS = [*T1_COLUMNS, "G2"]

TEMPORAL_FEATURE_SETS = {
    "T0_no_current_grades": T0_COLUMNS,
    "T1_after_first_grade": T1_COLUMNS,
    "T2_after_second_grade": T2_COLUMNS,
}


def load_student_performance_data(raw_data_dir: Path) -> pd.DataFrame:
    """Load and combine the math and Portuguese UCI student datasets.

    Parameters
    ----------
    raw_data_dir:
        Directory containing ``student-mat.csv`` and ``student-por.csv``.

    Returns
    -------
    pd.DataFrame
        Combined student performance data with ``subject`` and ``pass_fail``
        columns added.
    """
    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []

    for subject, filename in RAW_FILENAMES.items():
        path = raw_data_dir / filename
        if not path.exists():
            missing_files.append(str(path))
            continue

        frame = pd.read_csv(path, sep=";")
        frame.insert(0, "subject", subject)
        frames.append(frame)

    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing raw data files:\n{missing}")

    data = pd.concat(frames, ignore_index=True)
    data["pass_fail"] = (data["G3"] >= PASSING_GRADE).astype(int)
    return data


def split_feature_types(data: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names for a selected column list."""
    numeric_features = [
        column for column in columns if pd.api.types.is_numeric_dtype(data[column])
    ]
    categorical_features = [
        column for column in columns if not pd.api.types.is_numeric_dtype(data[column])
    ]
    return numeric_features, categorical_features

