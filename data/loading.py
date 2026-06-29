import pandas as pd
from config import DATA_PATH


#low_memory=False; forces pandas to read the whole file before deciding each column's type to avoid mixed types
#loads tab-separated values (TSV) file into a pandas DataFrame
def load_raw_data():
    df = pd.read_csv(DATA_PATH, sep="\t", low_memory=False)
    print(f"Loaded TCGA data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

#partitions the big table into one sub-table per distinct value in the Cancer Type column
def split_by_cancer_type(df):
    dfs = {}
    for cancer_type, sub_df in df.groupby("Cancer Type"):
        #drop=True throws the old index away instead of saving it as a column)
        sub_df = sub_df.drop(columns=["Cancer Type"]).reset_index(drop=True) #renumber the rows from 0
        dfs[cancer_type] = sub_df
    print(f"Split into {len(dfs)} cancer type datasets")
    return dfs


def load_data():
    df   = load_raw_data() #load the raw file
    dfs  = split_by_cancer_type(df) #split into cancer type datasets
    return dfs #return a dictionary of {cancer_type: DataFrame}
