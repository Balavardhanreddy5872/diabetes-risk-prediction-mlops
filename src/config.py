from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

TARGET = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PREDICTION_THRESHOLD = 0.50

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "diabetes_no_duplicates.csv"
BASELINE_RESULTS_PATH = REPORTS_DIR / "baseline_model_results.csv"
ADVANCED_RESULTS_PATH = REPORTS_DIR / "advanced_model_results.csv"
ALL_RESULTS_PATH = REPORTS_DIR / "all_model_comparison.csv"
FINAL_MODEL_METRICS_PATH = REPORTS_DIR / "final_model_metrics.json"
FINAL_MODEL_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "final_model_classification_report.json"
FINAL_MODEL_TEST_PREDICTIONS_PATH = REPORTS_DIR / "final_model_test_predictions.csv"

FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_MODEL_METADATA_PATH = MODELS_DIR / "final_model_metadata.json"

FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
