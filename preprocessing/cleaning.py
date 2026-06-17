import pandas as pd
from config import TARGET_ROW_RETENTION, SELECTED_DATASETS


def basic_clean(sub_df):
    """Drop fully empty, constant, and ID-like columns."""
    df = sub_df.copy()
    df = df.loc[:, df.notna().any()]                        # drop fully empty columns
    df = df.loc[:, df.nunique(dropna=False) > 1]           # drop constant columns
    df = df.loc[:, (df.nunique() / len(df)) < 0.95]        # drop ID-like columns
    return df


def maximize_complete_rows(sub_df, target_row_retention=TARGET_ROW_RETENTION):
    """
    Greedily drop columns with most missingness to maximize complete rows.
    Stops early when target row retention is achieved.
    """
    df = sub_df.copy()
    df = df.loc[:, df.notna().any()]

    missing_ratio = df.isna().mean().sort_values(ascending=False)

    best_df   = None
    best_rows = 0

    for col in missing_ratio.index:
        df            = df.drop(columns=[col])
        complete_rows = df.dropna().shape[0]

        if complete_rows > best_rows:
            best_rows = complete_rows
            best_df   = df.copy()

        if complete_rows >= target_row_retention * len(sub_df):
            break

    if best_df is not None:
        best_df = best_df.dropna()

    return best_df


def final_cleanup(df):
    """Remove any remaining constant columns after cleaning."""
    return df.loc[:, df.nunique() > 1]


def clean_datasets(dfs):
    """
    Full cleaning pipeline:
    1. basic_clean on all datasets
    2. maximize_complete_rows on all datasets
    3. final_cleanup on all datasets
    4. Keep only selected datasets with >10 rows
    """
    # step 1: basic clean
    dfs = {k: basic_clean(v) for k, v in dfs.items()}

    # step 2: maximize complete rows
    dfs_cleaned = {}
    for k, sub_df in dfs.items():
        cleaned = maximize_complete_rows(sub_df)
        if cleaned is not None and len(cleaned) > 10:
            dfs_cleaned[k] = cleaned

    # step 3: final cleanup
    dfs_cleaned = {k: final_cleanup(v) for k, v in dfs_cleaned.items()}

    # step 4: keep only selected datasets
    dfs_cleaned = {k: v for k, v in dfs_cleaned.items() if k in SELECTED_DATASETS}

    print(f"Cleaned {len(dfs_cleaned)} datasets")
    for name, df in dfs_cleaned.items():
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return dfs_cleaned
