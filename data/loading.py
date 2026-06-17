import pandas as pd
from config import DATA_FILE


def load_raw_data():
    """Load raw TCGA clinical data from GitHub."""
    df = pd.read_csv(DATA_FILE, sep="\t", low_memory=False)
    print(f"Loaded TCGA data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_by_cancer_type(df):
    """Split dataframe into sub-dataframes by Cancer Type column."""
    dfs = {}
    for cancer_type, sub_df in df.groupby("Cancer Type"):
        sub_df = sub_df.drop(columns=["Cancer Type"]).reset_index(drop=True)
        dfs[cancer_type] = sub_df
    print(f"Split into {len(dfs)} cancer type datasets")
    return dfs


def load_data():
    """Full data loading pipeline."""
    df   = load_raw_data()
    dfs  = split_by_cancer_type(df)
    return dfs
