import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, f_oneway, chi2_contingency
from config import MIN_GROUP_SIZE, MAX_CARDINALITY

#Split dataframe columns into numeric and categorical.
def split_columns(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    return num_cols, cat_cols

#Pearson correlation for all numeric column pairs.
def run_pearson(df, num_cols):
    results = []
    for col1, col2 in combinations(num_cols, 2): #yields every unordered pair of numeric columns 
        # take just those two columns, 
        # drop rows where either is missing, and skip the pair if fewer than 5 rows remain
        try:
            valid = df[[col1, col2]].dropna()
            if len(valid) < 5:
                continue
            #returns the correlation coefficient and a p-value, both stored
            corr, p = pearsonr(valid[col1], valid[col2])
            results.append({"var1": col1, "var2": col2, "correlation": corr, "p_value": p})
        except:
            continue
    return pd.DataFrame(results)

#One-way ANOVA for all numeric-categorical column pairs.
def run_anova(df, num_cols, cat_cols):
    results = []
    # loopover categorical columns, but skip ones that aren't usable as groupings: 
    # fewer than 2 categories (nothing to compare) or 
    # more than MAX_CARDINALITY (10) categories (too many groups, likely an ID-like column)
    for cat in cat_cols:
        if df[cat].nunique() < 2 or df[cat].nunique() > MAX_CARDINALITY:
            continue
        #For each numeric column paired with that category: drop missing rows
        # and skip if any group has fewer than MIN_GROUP_SIZE (5) observations
        for num in num_cols:
            try:
                valid  = df[[cat, num]].dropna()
                groups = valid.groupby(cat)[num].apply(list)
                if any(len(g) < MIN_GROUP_SIZE for g in groups):
                    continue
                f_stat, p = f_oneway(*groups)
                results.append({"categorical": cat, "numerical": num, "f_stat": f_stat, "p_value": p})
            except:
                continue
    return pd.DataFrame(results)

#Chi-Square test for all categorical column pairs.
def run_chi2(df, cat_cols):
    # crosstab builds the contingency table (counts of each category combination), 
    # and chi2_contingency tests whether the two variables are independent
    results = []
    for col1, col2 in combinations(cat_cols, 2):
        if df[col1].nunique() > MAX_CARDINALITY or df[col2].nunique() > MAX_CARDINALITY:
            continue
        try:
            valid = df[[col1, col2]].dropna()
            if len(valid) < 10:
                continue
            table = pd.crosstab(valid[col1], valid[col2])
            if table.size == 0:
                continue
            chi2, p, _, _ = chi2_contingency(table)
            results.append({"var1": col1, "var2": col2, "chi2": chi2, "p_value": p})
        except:
            continue
    return pd.DataFrame(results)

"""Pairwise as per literature: don't impute, just use whatever data is present for each calculation."""

#Pairwise Pearson; rename only for clean coding.
def run_pearson_pairwise(df, num_cols):
    results = []
    for col1, col2 in combinations(num_cols, 2):
        try:
            pair = df[[col1, col2]].dropna()
            if len(pair) < 5:
                continue
            corr, p = pearsonr(pair[col1], pair[col2])
            results.append({"var1": col1, "var2": col2, "correlation": corr, "p_value": p})
        except:
            continue
    return pd.DataFrame(results)

#Pairwise ANOVA; also identical to the non-pairwise version, but kept separate for clarity.
def run_anova_pairwise(df, num_cols, cat_cols):
    results = []
    for cat in cat_cols:
        if df[cat].nunique() < 2 or df[cat].nunique() > MAX_CARDINALITY:
            continue
        for num in num_cols:
            try:
                pair   = df[[cat, num]].dropna()
                groups = pair.groupby(cat)[num].apply(list)
                if any(len(g) < 5 for g in groups):
                    continue
                f_stat, p = f_oneway(*groups)
                results.append({"categorical": cat, "numerical": num, "f_stat": f_stat, "p_value": p})
            except:
                continue
    return pd.DataFrame(results)

#Pairwise Chi-Square; pairwise omits the if len(valid) < 10: continue guard
def run_chi2_pairwise(df, cat_cols):
    results = []
    for col1, col2 in combinations(cat_cols, 2):
        if df[col1].nunique() > MAX_CARDINALITY or df[col2].nunique() > MAX_CARDINALITY:
            continue
        try:
            pair  = df[[col1, col2]].dropna()
            table = pd.crosstab(pair[col1], pair[col2])
            if table.size == 0:
                continue
            chi2, p, _, _ = chi2_contingency(table)
            results.append({"var1": col1, "var2": col2, "chi2": chi2, "p_value": p})
        except:
            continue
    return pd.DataFrame(results)

#Run all three tests on a dataframe and return results dict.
def compute_stats(df):
    num_cols, cat_cols = split_columns(df)
    return {
        "pearson": run_pearson(df, num_cols),
        "anova":   run_anova(df, num_cols, cat_cols),
        "chi2":    run_chi2(df, cat_cols)
    }

#Run all three pairwise tests on a dataframe and return results dict.
def compute_pairwise_stats(df):
    num_cols, cat_cols = split_columns(df)
    return {
        "pearson": run_pearson_pairwise(df, num_cols),
        "anova":   run_anova_pairwise(df, num_cols, cat_cols),
        "chi2":    run_chi2_pairwise(df, cat_cols)
    }
