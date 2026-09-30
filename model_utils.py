"""Shared data validation and model utilities for the student-risk application."""
import re
import numpy as np
import pandas as pd

TARGET_CANDIDATES = ["At_Risk", "at_risk", "target", "Target", "risk_label"]
ID_CANDIDATES = ["Student_ID", "student_id", "id", "ID"]

def sequence_columns(df):
    pattern = re.compile(r"^week(\d+)_(.+)$")
    found = {}
    for col in df.columns:
        m = pattern.match(str(col))
        if m:
            found.setdefault(int(m.group(1)), []).append(col)
    weeks = sorted(found)
    if not weeks:
        raise ValueError(
            "No weekly sequence columns found. Expected columns like "
            "week1_attendance_pct, week1_lms_logins, ... week2_attendance_pct."
        )
    if weeks != list(range(1, max(weeks) + 1)):
        raise ValueError("Week columns must be consecutive, starting at week1.")
    feature_names = sorted(set(c.split("_", 1)[1] for cols in found.values() for c in cols))
    # Require a consistent set of features for each week.
    for w in weeks:
        expected = {f"week{w}_{name}" for name in feature_names}
        actual = set(found[w])
        if expected != actual:
            raise ValueError(f"Weekly features are inconsistent for week {w}.")
    cols_by_week = [[f"week{w}_{name}" for name in feature_names] for w in weeks]
    return feature_names, cols_by_week

def prepare_sequences(df, require_target=True):
    data = df.copy()
    target_col = next((c for c in TARGET_CANDIDATES if c in data.columns), None)
    if require_target and target_col is None:
        raise ValueError("Training CSV must contain an At_Risk target column (0=Safe, 1=At-Risk).")
    feature_names, cols_by_week = sequence_columns(data)
    ordered_cols = [c for week_cols in cols_by_week for c in week_cols]
    missing = data[ordered_cols].isna().sum().sum()
    if missing:
        data[ordered_cols] = data[ordered_cols].apply(pd.to_numeric, errors="coerce")
        data[ordered_cols] = data[ordered_cols].fillna(data[ordered_cols].median()).fillna(0)
    else:
        data[ordered_cols] = data[ordered_cols].apply(pd.to_numeric, errors="raise")
    x = data[ordered_cols].to_numpy(dtype=np.float32).reshape(
        len(data), len(cols_by_week), len(feature_names)
    )
    y = None
    if require_target:
        y = pd.to_numeric(data[target_col], errors="raise").to_numpy().astype(int)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Target must be binary: 0=Safe, 1=At-Risk.")
        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain both target classes.")
    return x, y, feature_names, cols_by_week, target_col

def risk_label(prob, threshold=0.5):
    return "At-Risk" if prob >= threshold else "Safe"
