# ============================================================
# missingness/simulation.py
# Simulates MCAR, MAR, MNAR missingness on cleaned datasets.
# ============================================================

import numpy as np
from missmecha.generator import MissMechaGenerator
from config import SELECTED_DATASETS, SEED


def restore_dtypes(df_missing, df_original):
    """
    Restore original dtypes after MissMechaGenerator converts everything to float.
    Starts from original dataframe and only punches in NaN positions from simulation.
    """
    df_fixed = df_original.copy()

    for col in df_original.columns:
        if col not in df_missing.columns:
            continue
        missing_mask = df_missing[col].isna()
        df_fixed.loc[missing_mask, col] = np.nan

    return df_fixed


def simulate_mcar(dfs_cleaned, missing_rate, seed=SEED):
    """Simulate Missing Completely At Random."""
    datasets = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        df       = dfs_cleaned[name].copy()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        mm       = MissMechaGenerator(
            mechanism="MCAR", mechanism_type=1,
            missing_rate=missing_rate, seed=seed, cat_cols=cat_cols
        )
        datasets[name] = mm.fit_transform(df)
        print(f"  MCAR {missing_rate} - {name} done")
    return datasets


def simulate_mar(dfs_cleaned, missing_rate, seed=SEED):
    datasets = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        df       = dfs_cleaned[name].copy()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        mm       = MissMechaGenerator(
            mechanism="MAR", mechanism_type=2,
            missing_rate=missing_rate, seed=seed, cat_cols=cat_cols
        )
        datasets[name] = mm.fit_transform(df)  # ← add this
        print(f"  MAR {missing_rate} - {name} done")
    return datasets


def simulate_mnar(dfs_cleaned, missing_rate, seed=SEED):
    datasets = {}
    for name in SELECTED_DATASETS:
        if name not in dfs_cleaned:
            continue
        df       = dfs_cleaned[name].copy()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        mm       = MissMechaGenerator(
            mechanism="MNAR", mechanism_type=1,
            missing_rate=missing_rate, seed=seed, cat_cols=cat_cols
        )
        datasets[name] = mm.fit_transform(df)  # ← add this
        print(f"  MNAR {missing_rate} - {name} done")
    return datasets


def simulate_all(dfs_cleaned, missing_rate, seed=SEED):
    """Simulate all three mechanisms for a given missing rate."""
    print(f"\nSimulating missingness at rate={missing_rate}")
    return {
        "MCAR": simulate_mcar(dfs_cleaned, missing_rate, seed),
        "MAR":  simulate_mar(dfs_cleaned,  missing_rate, seed),
        "MNAR": simulate_mnar(dfs_cleaned, missing_rate, seed)
    }
