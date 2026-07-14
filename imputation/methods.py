# ============================================================
# imputation/methods.py
# All imputation methods in one file.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from missforest import MissForest
from lightgbm import LGBMClassifier, LGBMRegressor
from analysis.tests import split_columns
from config import KNN_K, MICE_ITER, RF_TREES, SEED

#Remove all rows with any missing values.
def impute_complete_case(df):
    return df.dropna()

#Mean imputation for numeric, mode imputation for categorical.
def mean_mode_impute(df):
    df_imputed         = df.copy()
    num_cols, cat_cols = split_columns(df)

    if num_cols:
        mean_imputer          = SimpleImputer(strategy="mean")
        df_imputed[num_cols]  = mean_imputer.fit_transform(df[num_cols])

    if cat_cols:
        mode_imputer          = SimpleImputer(strategy="most_frequent")
        df_imputed[cat_cols]  = mode_imputer.fit_transform(df[cat_cols])

    return df_imputed

#Encode categorical columns as integers for imputation.
def _encode_categoricals(df_imputed, cat_cols):
    encoders = {}
    for col in cat_cols:
        le       = LabelEncoder() # LabelEncoder learns the mapping from category labels to integers
        non_null = df_imputed[col].dropna()
        le.fit(non_null)
        # so the same mapping can be reversed later 
        encoders[col]       = le
        df_imputed[col]     = df_imputed[col].map(
        #Rewrites the column: each real label becomes its integer code, and each NaN stays NaN
            lambda x: le.transform([x])[0] if pd.notna(x) else np.nan 
        )
    return df_imputed, encoders

#Decode categorical columns back to original string labels.
def _decode_categoricals(df_imputed, cat_cols, encoders):
    for col, le in encoders.items():
        if col not in df_imputed.columns:
            continue
        #if any NaN somehow survived imputation,
        #fill it with the column's most common value (mode()[0]) so the next steps don't choke on NaN.
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
        # The imputed floats get snapped back to valid category codes
        df_imputed[col] = df_imputed[col].round().clip(0, len(le.classes_) - 1).astype(int)
        #saved encoder reverses the mapping: integers go back to their original string labels
        df_imputed[col] = le.inverse_transform(df_imputed[col])
    return df_imputed

#KNN imputation — finds K nearest neighbours and fills missing values.
def knn_impute(df, k=KNN_K):
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

#Single-imputation MICE variant using sklearn IterativeImputer
# with BayesianRidge (default estimator).
def mice_impute(df, max_iter=MICE_ITER):
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

# MissForest imputation using the dedicated MissForest package
# Categorical columns are label-encoded to integer codes and named via `categorical=`
# so the package fits a classifier (LGBMClassifier) on them and a regressor (LGBMRegressor) on the
# numeric columns: a genuine mixed MissForest model rather than regression with rounding. 
# RF_TREES maps to each learner's number of trees (n_estimators) and
# MICE_ITER to the number of imputation iterations (max_iter).
def missforest_impute(df, n_estimators=RF_TREES, max_iter=MICE_ITER):
    df_imputed         = df.copy()
    num_cols, cat_cols = split_columns(df)

    # The package needs numeric input and identifies categorical targets by name,
    # so encode categoricals to integer codes (NaNs preserved) first.
    df_imputed, encoders = _encode_categoricals(df_imputed, cat_cols)

    # LightGBM rejects feature names containing special JSON characters (which
    # the clinical column names contain), so temporarily rename columns to safe
    # placeholders and restore the originals afterwards.

    #enumerate is a built-in Python function that, when you loop over a collection, 
    #gives you both a running counter and each item

    #Build the safe-name map
    #e.g:- {"Diagnosis Age": "f0", "Tumor Stage": "f1"}
    safe_names = {col: f"f{i}" for i, col in enumerate(df_imputed.columns)}

    #Build the reverse map so we can restore the original names later
    #e.g:- {"f0": "Diagnosis Age", "f1": "Tumor Stage"}
    orig_names = {v: k for k, v in safe_names.items()} 

    #Rename the columns to safe names for LightGBM
    # (f0, f1, …)
    df_imputed = df_imputed.rename(columns=safe_names)
    #Translate the categorical list
    #["Tumor Stage", "Grade"] to ["f1", "f5"]
    cat_safe   = [safe_names[c] for c in cat_cols]

    imputer = MissForest(
        # the model used for categorical columns
        clf=LGBMClassifier(n_estimators=n_estimators, random_state=SEED, verbosity=-1), #verbosity=-1 to silence LightGBM's logging
        # the model for numeric columns 
        rgr=LGBMRegressor(n_estimators=n_estimators, random_state=SEED, verbosity=-1),
        categorical=cat_safe if cat_safe else None,
        max_iter=max_iter,
        verbose=0,
    )
    #learns from the observed values and returns a new DataFrame with every missing cell filled.
    df_imputed = imputer.fit_transform(df_imputed)
    df_imputed = df_imputed.rename(columns=orig_names)

    # The package reorders columns internally; restore the original row and
    # column order so downstream alignment stays correct.
    df_imputed = df_imputed.reindex(index=df.index, columns=df.columns)

    for col in num_cols:
        df_imputed[col] = pd.to_numeric(df_imputed[col], errors="coerce")

    df_imputed = _decode_categoricals(df_imputed, cat_cols, encoders)
    return df_imputed

#Run all imputation methods on all mechanism datasets.
#Returns nested dict: imputed_datasets[method][mech][name]
def run_all_imputations(mechanisms):
    
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
                print(f"  {method_name} | {mech} | {name} - successfully imputed")

    return imputed_datasets