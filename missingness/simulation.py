import numpy as np
from missmecha.generator import MissMechaGenerator
from config import SELECTED_DATASETS, SEED

#Simulate Missing Completely At Random.
#fit_transform looks at clean dataframe to plan where missingness should go 
def simulate_mcar(dfs_cleaned, missing_rate, seed=SEED):
    datasets = {}
    for name in SELECTED_DATASETS:
        #Loop over datasets
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

#Simulate Missing At Random.
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
        datasets[name] = mm.fit_transform(df) 
        print(f"  MAR {missing_rate} - {name} done")
    return datasets

#Simulate Missing Not At Random.
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
        datasets[name] = mm.fit_transform(df)  
        print(f"  MNAR {missing_rate} - {name} done")
    return datasets

# Simulate all three mechanisms for a given missing rate.
def simulate_all(dfs_cleaned, missing_rate, seed=SEED):
    print(f"\nSimulating missingness at rate = {missing_rate}")
    return {
        "MCAR": simulate_mcar(dfs_cleaned, missing_rate, seed),
        "MAR":  simulate_mar(dfs_cleaned,  missing_rate, seed),
        "MNAR": simulate_mnar(dfs_cleaned, missing_rate, seed)
    }
