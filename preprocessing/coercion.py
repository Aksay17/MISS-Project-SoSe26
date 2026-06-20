# ============================================================
# preprocessing/coercion.py
# Fixes misclassified column types after cleaning.
# ============================================================

import pandas as pd
from config import SELECTED_DATASETS


def detect_suspected_numeric(dfs_cleaned):
    """
    Detect object columns that are actually numeric
    (stored as strings but contain numbers).
    """
    suspected_numeric = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        df   = dfs_cleaned[name]
        cols = []
        for col in df.select_dtypes(include="object").columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / len(df) > 0.8:
                cols.append(col)
        suspected_numeric[name] = cols
    return suspected_numeric


def detect_suspected_categorical(dfs_cleaned):
    """
    Detect numeric columns that are actually categorical
    (stored as ints but represent categories like 0/1 flags).
    """
    suspected_categorical = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        df   = dfs_cleaned[name]
        cols = []
        for col in df.select_dtypes(include="number").columns:
            if df[col].nunique() <= 10 and df[col].dropna().apply(float.is_integer).all():
                cols.append(col)
        suspected_categorical[name] = cols
    return suspected_categorical


def coerce_types(dfs_cleaned):
    """
    Apply type coercions:
    - Convert suspected categorical columns to string
    - Convert suspected numeric columns to float, drop if NaNs introduced
    """
    suspected_numeric     = detect_suspected_numeric(dfs_cleaned)
    suspected_categorical = detect_suspected_categorical(dfs_cleaned)

    # convert suspected categorical → string
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        sub_df = dfs_cleaned[name].copy()
        for col in suspected_categorical[name]:
            sub_df[col] = sub_df[col].astype(str)
        dfs_cleaned[name] = sub_df

    # convert suspected numeric → float, drop column if NaNs introduced
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        sub_df = dfs_cleaned[name].copy()
        for col in suspected_numeric[name]:
            converted = pd.to_numeric(sub_df[col], errors="coerce")
            if converted.isna().any():
                sub_df = sub_df.drop(columns=[col])
            else:
                sub_df[col] = converted
        dfs_cleaned[name] = sub_df

    # verify no NaNs remain
    for name, df in dfs_cleaned.items():
        assert df.isna().sum().sum() == 0, f"{name} has NaNs after type coercion"

    print("Type coercion complete — all datasets have 0 NaNs")
    return dfs_cleaned
