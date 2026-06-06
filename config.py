# ============================================================
# config.py
# All project-wide settings in one place.
# Change values here to re-run with different parameters.
# ============================================================
DATA_FILE = r"C:\Users\PC\Documents\GitHub\MISS-Project-SoSe26\data\data_file\cBioportal_data.tsv"
# Dataset
SELECTED_DATASETS = [
    "Invasive Breast Carcinoma",
    "Non-Small Cell Lung Cancer",
    "Well-Differentiated Thyroid Cancer",
    "Endometrial Carcinoma"
]

# Cleaning
TARGET_ROW_RETENTION = 0.2

# Missingness simulation
MISSING_RATES = [0.1, 0.2, 0.3]   # add/remove rates here to test more
SEED          = 42

# Imputation
KNN_K      = 5
MICE_ITER  = 10
RF_TREES   = 100

# Statistics
ALPHA          = 0.05   # significance threshold for p-value agreement
MIN_GROUP_SIZE = 5      # minimum group size for ANOVA
MAX_CARDINALITY = 10    # maximum unique values for categorical tests

# Output
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
