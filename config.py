# Dataset path, used to access data file in the data folder. 
# used in data/loading.py and regenerate_plots.py
DATA_PATH = r"ADD LOCAL DATA PATH HERE"

# selected list of datasets to be used in the analysis 
# used in missingness/simulation.py, preprocessing/cleaning.py, preprocessing/coercion.py
SELECTED_DATASETS = [
    "Invasive Breast Carcinoma",
    "Non-Small Cell Lung Cancer",
    "Well-Differentiated Thyroid Cancer",
    "Endometrial Carcinoma"
]

# Cleaning, decision to drop rows with too many missing values
# controls the trade-off when deciding which columns to drop
# used only in preprocessing/cleaning.py
TARGET_ROW_RETENTION = 0.2

# Missingness simulation, used in main.py to loop over missing rates and generate missingness
# Each value triggers a full simulate - impute - test cycle, 
MISSING_RATES = [0.1 , 0.2, 0.3]  

# default random seed used in missingness/simulation.py, MICE & MissForest Imputation 
SEED = 42

# Imputation
KNN_K      = 5 # the number of neighbors KNN averages over to fill a missing cell
MICE_ITER  = 10 # controls number of iterations for MICE & MISSForest imputation
RF_TREES   = 100 #  default n_estimators for the random forest, i.e., how many trees per forest

# Statistics
MIN_GROUP_SIZE = 5 # minimum size for ANOVA a categorical-vs-numeric comparison is skipped if any group has fewer than 5 observations


MAX_CARDINALITY = 10 # maximum unique values for categorical tests
#  It skips categorical columns with more than 10 unique values 
# to avoid treating near-continuous or ID-like columns as categories

# Output
PLOTS_DIR   = "plots"
