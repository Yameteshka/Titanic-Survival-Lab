import argparse
import json
import logging
import random
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from code.config import (
    METADATA_PATH, MODEL_PATH, MODELS_DIR, PREPROCESSOR_PATH, RANDOM_SEED, TARGET,
    TEST_DATA_PATH, THRESHOLD, TRAIN_DATA_PATH,
)
from code.models.features import build_preprocessor
from code.models.model import TitanicLinearClassifier

LOGGER = logging.getLogger(__name__)


def set_deterministic_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_model(epochs: int = 250, learning_rate: float = 0.01) -> dict:
    if not TRAIN_DATA_PATH.is_file() or not TEST_DATA_PATH.is_file():
        raise FileNotFoundError("Processed train/test files are missing. Run Stage 1 first.")
    set_deterministic_seed()
    train = pd.read_csv(TRAIN_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(np.float32)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_train).astype(np.float32)
    feature_names = preprocessor.named_steps["columns"].get_feature_names_out().tolist()

    model = TitanicLinearClassifier(transformed.shape[1])
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_tensor = torch.from_numpy(transformed)
    y_tensor = torch.from_numpy(y_train.to_numpy()).reshape(-1, 1)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_function(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            LOGGER.info("Epoch %d/%d - training loss %.6f", epoch, epochs, loss.item())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": transformed.shape[1]}, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    metadata = {
        "model_type": "PyTorch single-neuron logistic classifier (nn.Linear(input_dim, 1))",
        "input_feature_count": int(transformed.shape[1]),
        "feature_names": feature_names,
        "raw_features": X_train.columns.tolist(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "classification_threshold": THRESHOLD,
        "dataset_sizes": {"train": int(len(train)), "test": int(len(test))},
        "random_seed": RANDOM_SEED,
        "model_version": datetime.now(timezone.utc).strftime("titanic-%Y%m%d%H%M%S"),
        "training": {"epochs": epochs, "learning_rate": learning_rate, "optimizer": "Adam", "loss": "BCEWithLogitsLoss"},
        "artifacts": {"model": MODEL_PATH.name, "preprocessor": PREPROCESSOR_PATH.name, "metrics": "metrics.json"},
        "versions": {"torch": torch.__version__, "scikit_learn": sklearn.__version__},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved model, fitted preprocessor, and metadata to %s", MODELS_DIR)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_model(args.epochs, args.learning_rate)
