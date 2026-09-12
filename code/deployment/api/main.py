import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from code.config import METADATA_PATH, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH
from code.deployment.api.schemas import PassengerInput, PredictionOutput
from code.models.inference import TitanicPredictor


def _artifact_path(env_name: str, default: Path) -> Path:
    return Path(os.getenv(env_name, str(default)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = TitanicPredictor(
        model_path=_artifact_path("MODEL_PATH", MODEL_PATH),
        preprocessor_path=_artifact_path("PREPROCESSOR_PATH", PREPROCESSOR_PATH),
        metadata_path=_artifact_path("METADATA_PATH", METADATA_PATH),
        metrics_path=_artifact_path("METRICS_PATH", METRICS_PATH),
    )
    yield


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    predictor = request.app.state.predictor
    return {"status": "ok", "model_loaded": True, "model_version": predictor.metadata["model_version"], "trained_at": predictor.metadata["trained_at"]}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    predictor = request.app.state.predictor
    return {
        "model_type": predictor.metadata["model_type"],
        "model_version": predictor.metadata["model_version"],
        "trained_at": predictor.metadata["trained_at"],
        "dataset_sizes": predictor.metadata["dataset_sizes"],
        "raw_features": predictor.metadata["raw_features"],
        "feature_names": predictor.metadata["feature_names"],
        "test_metrics": predictor.metrics,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput, request: Request) -> dict:
    frame = pd.DataFrame([passenger.as_model_record()])
    return request.app.state.predictor.predict_frame(frame)[0]


def _validate_records(records: list[dict]) -> list[PassengerInput]:
    validated, errors = [], []
    for index, record in enumerate(records):
        normalized = {str(key).lower(): value for key, value in record.items()}
        try:
            validated.append(PassengerInput.model_validate(normalized))
        except ValidationError as error:
            errors.append({"row": index + 1, "errors": error.errors(include_url=False)})
    if errors:
        raise HTTPException(status_code=422, detail={"message": "Batch validation failed", "rows": errors})
    if not validated:
        raise HTTPException(status_code=422, detail="Batch is empty")
    return validated


@app.post("/predict-batch")
async def predict_batch(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not getattr(upload, "filename", "").lower().endswith(".csv"):
            raise HTTPException(status_code=422, detail="Upload a CSV file in the 'file' field")
        try:
            records = pd.read_csv(io.BytesIO(await upload.read())).to_dict(orient="records")
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"Could not parse CSV: {error}") from error
    else:
        payload = await request.json()
        records = payload.get("passengers", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise HTTPException(status_code=422, detail="Expected a list or {'passengers': [...]} JSON")
    passengers = _validate_records(records)
    frame = pd.DataFrame([passenger.as_model_record() for passenger in passengers])
    return {"count": len(passengers), "predictions": request.app.state.predictor.predict_frame(frame)}
