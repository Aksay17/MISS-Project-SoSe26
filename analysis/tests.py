# ============================================================
# statistics/tests.py
# Statistical tests: Pearson, ANOVA, Chi-Square.
# ============================================================

import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, f_oneway, chi2_contingency
from config import MIN_GROUP_SIZE, MAX_CARDINALITY

#Split dataframe columns into numeric and categorical.
def split_columns(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    return num_cols, cat_cols

#Pearson correlation for all numeric column pairs
def run_pearson(df, num_cols):
    results = []
    for col1, col2 in combinations(num_cols, 2):
        try:
            valid = df[[col1, col2]].dropna()
            if len(valid) < 5:
                continue
            corr, p = pearsonr(valid[col1], valid[col2])
            results.append({"var1": col1, "var2": col2, "correlation": corr, "p_value": p})
        except:
            continue
    return pd.DataFrame(results)

#One-way ANOVA for all numeric-categorical column pairs.
def run_anova(df, num_cols, cat_cols):
    results = []
    for cat in cat_cols:
        if df[cat].nunique() < 2 or df[cat].nunique() > MAX_CARDINALITY:
            continue
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


def run_chi2(df, cat_cols):
    """Chi-Square test for all categorical column pairs."""
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


def run_pearson_pairwise(df, num_cols):
    """Pairwise Pearson — drops NaNs per pair."""
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


def run_anova_pairwise(df, num_cols, cat_cols):
    """Pairwise ANOVA — drops NaNs per pair."""
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


def run_chi2_pairwise(df, cat_cols):
    """Pairwise Chi-Square — drops NaNs per pair."""
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


def compute_stats(df):
    """Run all three tests on a dataframe and return results dict."""
    num_cols, cat_cols = split_columns(df)
    return {
        "pearson": run_pearson(df, num_cols),
        "anova":   run_anova(df, num_cols, cat_cols),
        "chi2":    run_chi2(df, cat_cols)
    }


def compute_pairwise_stats(df):
    """Run all three pairwise tests on a dataframe and return results dict."""
    num_cols, cat_cols = split_columns(df)
    return {
        "pearson": run_pearson_pairwise(df, num_cols),
        "anova":   run_anova_pairwise(df, num_cols, cat_cols),
        "chi2":    run_chi2_pairwise(df, cat_cols)
    }
