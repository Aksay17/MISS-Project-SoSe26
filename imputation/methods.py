# ============================================================
# imputation/methods.py
# All imputation methods in one file.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from analysis.tests import split_columns
from config import KNN_K, MICE_ITER, RF_TREES, SEED


def impute_complete_case(df):
    """Remove all rows with any missing values."""
    return df.dropna()


def mean_mode_impute(df):
    """Mean imputation for numeric, mode imputation for categorical."""
    df_imputed         = df.copy()
    num_cols, cat_cols = split_columns(df)

    if num_cols:
        mean_imputer          = SimpleImputer(strategy="mean")
        df_imputed[num_cols]  = mean_imputer.fit_transform(df[num_cols])

    if cat_cols:
        mode_imputer          = SimpleImputer(strategy="most_frequent")
        df_imputed[cat_cols]  = mode_imputer.fit_transform(df[cat_cols])

    return df_imputed


def _encode_categoricals(df_imputed, cat_cols):
    """Encode categorical columns as integers for imputation."""
    encoders = {}
    for col in cat_cols:
        le       = LabelEncoder()
        non_null = df_imputed[col].dropna()
        le.fit(non_null)
        encoders[col]       = le
        df_imputed[col]     = df_imputed[col].map(
            lambda x: le.transform([x])[0] if pd.notna(x) else np.nan
        )
    return df_imputed, encoders


def _decode_categoricals(df_imputed, cat_cols, encoders):
    """Decode categorical columns back to original string labels."""
    for col, le in encoders.items():
        if col not in df_imputed.columns:
            continue
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
        df_imputed[col] = df_imputed[col].round().clip(0, len(le.classes_) - 1).astype(int)
        df_imputed[col] = le.inverse_transform(df_imputed[col])
    return df_imputed


def knn_impute(df, k=KNN_K):
    """KNN imputation — finds K nearest neighbours and fills missing values."""
    df_imputed         = df.copy()
    num_cols, cat_cols = split_columns(df)

    df_imputed, encoders = _encode_categoricals(df_imputed, cat_cols)

    imputer    = KNNImputer(n_neighbors=k)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_imputed),
        columns=df_imputed.columns
    )

    # restore numeric dtypes lost during DataFrame reconstruction
    for col in num_cols:
        df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

    df_imputed = _decode_categoricals(df_imputed, cat_cols, encoders)
    return df_imputed


def mice_impute(df, max_iter=MICE_ITER):
    """
    Single-imputation MICE variant using sklearn IterativeImputer
    with BayesianRidge (default estimator).
    Based on: van Buuren & Groothuis-Oudshoorn (2011).
    """
    df_imputed         = df.copy()
    num_cols, cat_cols = split_columns(df)

    df_imputed, encoders = _encode_categoricals(df_imputed, cat_cols)

    imputer    = IterativeImputer(max_iter=max_iter, random_state=SEED)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_imputed),
        columns=df_imputed.columns
    )

    for col in num_cols:
        df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

    df_imputed = _decode_categoricals(df_imputed, cat_cols, encoders)
    return df_imputed


def missforest_impute(df, n_estimators=RF_TREES, max_iter=MICE_ITER):
    """
    MissForest imputation using IterativeImputer with RandomForestRegressor.
    Based on: Stekhoven & Bühlmann (2012).
    """
    df_imputed         = df.copy()
    num_cols, cat_cols = split_columns(df)

    df_imputed, encoders = _encode_categoricals(df_imputed, cat_cols)

    imputer    = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=SEED),
        max_iter=max_iter,
        random_state=SEED
    )
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_imputed),
        columns=df_imputed.columns
    )

    for col in num_cols:
        df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

    df_imputed = _decode_categoricals(df_imputed, cat_cols, encoders)
    return df_imputed


def run_all_imputations(mechanisms):
    """
    Run all imputation methods on all mechanism datasets.
    Returns nested dict: imputed_datasets[method][mech][name]
    """
    imputation_fns = {
        "complete_case": impute_complete_case, 
        "mean_mode":  mean_mode_impute,
        "knn":        knn_impute,
        "mice":       mice_impute,
        "missforest": missforest_impute
    }

    imputed_datasets = {}

    for method_name, impute_fn in imputation_fns.items():
        imputed_datasets[method_name] = {}
        for mech, datasets in mechanisms.items():
            imputed_datasets[method_name][mech] = {}
            for name, df in datasets.items():
                imputed_datasets[method_name][mech][name] = impute_fn(df)
                print(f"  {method_name} | {mech} | {name} done")

    return imputed_datasets
