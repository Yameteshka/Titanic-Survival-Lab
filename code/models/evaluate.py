import json
import logging
import os

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score, confusion_matrix

from code.config import (
    CLEANING_REPORT_PATH, METADATA_PATH, METRICS_PATH, MLRUNS_DIR, MODEL_PATH,
    PREPROCESSOR_PATH, RANDOM_SEED, TARGET, TEST_DATA_PATH,
)
from code.models.inference import TitanicPredictor

LOGGER = logging.getLogger(__name__)


def evaluate_and_log() -> dict:
    test = pd.read_csv(TEST_DATA_PATH)
    predictor = TitanicPredictor(metrics_path=METRICS_PATH)
    predictions = predictor.predict_frame(test.drop(columns=[TARGET]))
    y_true = test[TARGET].astype(int).to_numpy()
    probabilities = [row["probability_survived"] for row in predictions]
    classes = [row["predicted_class"] for row in predictions]
    matrix = confusion_matrix(y_true, classes).tolist()
    metrics = {
        "dataset": "test",
        "test_rows": int(len(test)),
        "accuracy": float(accuracy_score(y_true, classes)),
        "precision": float(precision_score(y_true, classes, zero_division=0)),
        "recall": float(recall_score(y_true, classes, zero_division=0)),
        "f1": float(f1_score(y_true, classes, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "confusion_matrix": {"labels": [0, 1], "values": matrix},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("TEST metrics: %s", json.dumps(metrics))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (MLRUNS_DIR / "local").resolve().as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Titanic Survival MLOps")
    metadata = predictor.metadata
    with mlflow.start_run(run_name=metadata["model_version"]):
        mlflow.log_params({
            "epochs": metadata["training"]["epochs"],
            "learning_rate": metadata["training"]["learning_rate"],
            "optimizer": "Adam",
            "model_type": "nn.Linear(input_dim, 1)",
            "threshold": metadata["classification_threshold"],
            "train_size": metadata["dataset_sizes"]["train"],
            "test_size": metadata["dataset_sizes"]["test"],
            "random_seed": RANDOM_SEED,
        })
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
        for artifact in [METRICS_PATH, METADATA_PATH, CLEANING_REPORT_PATH, MODEL_PATH, PREPROCESSOR_PATH]:
            mlflow.log_artifact(str(artifact), artifact_path="pipeline_artifacts")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_and_log()
