import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from code.config import METADATA_PATH, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH
from code.models.model import TitanicLinearClassifier


class TitanicPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        metadata_path: Path = METADATA_PATH,
        metrics_path: Path = METRICS_PATH,
    ) -> None:
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        self.preprocessor = joblib.load(preprocessor_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = TitanicLinearClassifier(int(checkpoint["input_dim"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(self.metadata["classification_threshold"])

    def predict_frame(self, frame: pd.DataFrame) -> list[dict]:
        transformed = self.preprocessor.transform(frame).astype(np.float32)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(torch.from_numpy(transformed))).numpy().ravel()
        return [
            {
                "predicted_class": int(probability >= self.threshold),
                "probability_survived": float(probability),
                "probability_not_survived": float(1.0 - probability),
                "threshold": self.threshold,
                "model_version": self.metadata["model_version"],
            }
            for probability in probabilities
        ]
