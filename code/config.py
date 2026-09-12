from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pt"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"
RANDOM_SEED = 42
THRESHOLD = 0.5
