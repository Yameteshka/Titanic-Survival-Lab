import pytest

from code.config import METADATA_PATH, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH
from code.datasets.preprocess import clean_and_split_data
from code.models.evaluate import evaluate_and_log
from code.models.train import train_model


@pytest.fixture(scope="session", autouse=True)
def pipeline_artifacts() -> None:
    if not all(path.is_file() for path in [MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH, METRICS_PATH]):
        clean_and_split_data()
        train_model(epochs=30)
        evaluate_and_log()
