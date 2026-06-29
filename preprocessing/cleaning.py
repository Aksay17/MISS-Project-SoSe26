import pandas as pd
from config import TARGET_ROW_RETENTION, SELECTED_DATASETS


def basic_clean(sub_df):
    df = sub_df.copy()
    # isTrue column that has at least one non-null value, so fully empty columns get dropped.
    df = df.loc[:, df.notna().any()]  
    #keeps columns with more than one distinct value, NaNs treated as a distinct value, so constant columns get dropped.
    df = df.loc[:, df.nunique(dropna=False) > 1] 
    df = df.loc[:, (df.nunique() / len(df)) < 0.95]   # drop ID-like columns     
    return df

# Greedily drop columns with most missingness to maximize complete rows.
# Stops early when target row retention is achieved.
def maximize_complete_rows(sub_df, target_row_retention=TARGET_ROW_RETENTION):
    df = sub_df.copy()
    df = df.loc[:, df.notna().any()] # drop any fully-empty column

    #gives the missing fraction per column, sorted descending
    missing_ratio = df.isna().mean().sort_values(ascending=False)

    best_df   = None
    best_rows = 0

    for col in missing_ratio.index:
        df            = df.drop(columns=[col]) #drops the column with the most missingness
        complete_rows = df.dropna().shape[0] # counts the number of complete rows

        if complete_rows > best_rows: #check if this is the best we've seen so far
            best_rows = complete_rows #update the best number of complete rows
            best_df   = df.copy()

        if complete_rows >= target_row_retention * len(sub_df): #stop early if we have enough complete rows
            break

    if best_df is not None:
        best_df = best_df.dropna() #remove any remaining rows with missing values to ensure a fully complete dataset

    return best_df


def final_cleanup(df):
    return df.loc[:, df.nunique() > 1] #Remove any remaining constant columns after cleaning.

# orchestrates the four steps over every dataset
def clean_datasets(dfs):
    # step 1: basic clean (can be improved logic-wise because wasted effort)
    dfs = {k: basic_clean(v) for k, v in dfs.items()}

    # step 2: maximize complete rows, loops for each dataset 
    dfs_cleaned = {}
    for k, sub_df in dfs.items():
        cleaned = maximize_complete_rows(sub_df)
        #decides whether a freshly-cleaned dataset is good enough to keep
        if cleaned is not None and len(cleaned) > 10: #did cleaning actually produce a table? does the surviving table have more than 10 rows?
            dfs_cleaned[k] = cleaned

    # step 3: final cleanup
    dfs_cleaned = {k: final_cleanup(v) for k, v in dfs_cleaned.items()}

    # step 4: keep only selected datasets
    dfs_cleaned = {k: v for k, v in dfs_cleaned.items() if k in SELECTED_DATASETS}

    print(f"Cleaned {len(dfs_cleaned)} datasets")
    for name, df in dfs_cleaned.items():
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    return dfs_cleaned
