import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
STAGING_DATA_DIR = DATA_DIR / "staging"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
ANALYTICAL_DATA_DIR = DATA_DIR / "analytical"

DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Ensure all primary directories exist
for path in [
    RAW_DATA_DIR,
    STAGING_DATA_DIR,
    CLEANED_DATA_DIR,
    ANALYTICAL_DATA_DIR,
    DATABASE_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Primary Source Dataset Path
SOURCE_CSV = BASE_DIR / "data.csv"

# SQLite & PostgreSQL Database Config
# By default, use local SQLite database file; can be overridden by POSTGRES_URL
DB_PATH = DATABASE_DIR / "student_warehouse.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DATABASE_DIR}/mlflow.db")
MLFLOW_EXPERIMENT_NAME = "student_dropout_prediction"

# API Settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Model Hyperparameters & Features
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

# Drift Monitoring Thresholds (Kolmogorov-Smirnov test p-value & PSI)
DRIFT_P_VALUE_THRESHOLD = 0.05
DRIFT_PSI_THRESHOLD = 0.25
