import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk_platform.api.schemas import (
    CustomDatasetTrainRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from credit_risk_platform.config.datasets import DatasetConfig, register_dataset
from credit_risk_platform.config.settings import get_settings
from credit_risk_platform.database.init_db import init_db
from credit_risk_platform.database.models import PredictionHistory
from credit_risk_platform.database.session import get_db
from credit_risk_platform.feature_engineering.schema import infer_ordinal_mappings, profile_schema
from credit_risk_platform.models.registry import load_model_bundle
from credit_risk_platform.services.prediction import score_applicant

CUSTOM_DATASET_DIR = Path("data/raw/uploads")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(title="Credit Risk Decision Platform", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", model_loaded=settings.model_artifact_path.exists())


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return score_applicant(payload.features, db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...), db: Session = Depends(get_db)) -> list[dict]:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    frame = pd.read_csv(file.file)
    return [score_applicant(row.dropna().to_dict(), db) for _, row in frame.iterrows()]


@app.post("/custom-datasets/upload")
async def upload_custom_dataset(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    frame = pd.read_csv(file.file)
    if frame.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    dataset_id = f"custom_{uuid4().hex[:8]}"
    upload_path = CUSTOM_DATASET_DIR / f"{dataset_id}.csv"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(upload_path, index=False)

    ordinal_mappings = infer_ordinal_mappings(frame)
    profile = profile_schema(frame, ordinal_mappings)
    type_by_column: dict[str, str] = {}
    for column in profile.date_columns:
        type_by_column[column] = "date"
    for column in profile.boolean_columns:
        type_by_column[column] = "boolean"
    for column in profile.ordinal_columns:
        type_by_column[column] = "ordinal"
    for column in profile.numeric_columns:
        type_by_column[column] = "numeric"
    for column in profile.categorical_columns:
        type_by_column[column] = "categorical"
    for column in profile.dropped_columns:
        type_by_column[column] = "dropped"

    columns = []
    for column in frame.columns:
        uniques = frame[column].dropna().unique()
        columns.append(
            {
                "name": column,
                "type": type_by_column.get(column, "categorical"),
                "unique_values": (
                    [str(value) for value in uniques[:10]] if len(uniques) <= 10 else None
                ),
            }
        )
    return {"dataset_id": dataset_id, "row_count": len(frame), "columns": columns}


@app.post("/custom-datasets/{dataset_id}/train")
def train_custom_dataset(dataset_id: str, payload: CustomDatasetTrainRequest) -> dict:
    from credit_risk_platform.training.train import train_experiment

    upload_path = CUSTOM_DATASET_DIR / f"{dataset_id}.csv"
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")

    frame = pd.read_csv(upload_path)
    if payload.target_column not in frame.columns:
        raise HTTPException(status_code=400, detail=f"Column '{payload.target_column}' not found.")

    target = frame[payload.target_column].dropna()
    uniques = target.unique().tolist()
    if len(uniques) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Target column must have exactly 2 unique values (found {len(uniques)}).",
        )

    already_binary = set(map(str, uniques)) <= {"0", "1"}
    if not already_binary:
        if not payload.positive_label:
            raise HTTPException(
                status_code=400,
                detail="positive_label is required: which value represents the default/bad outcome?",  # noqa: E501
            )
        if payload.positive_label not in map(str, uniques):
            raise HTTPException(
                status_code=400, detail="positive_label does not match either value in the column."
            )
        # Normalize the target to clean 0/1 ourselves and persist it, so the
        # existing pipeline's unconditional `.astype(int)` (training/data.py)
        # just works with target_mapping=None, avoiding dtype-matching edge
        # cases in building a target_mapping dict against the column's raw dtype.
        frame[payload.target_column] = frame[payload.target_column].apply(
            lambda value: 1 if str(value) == payload.positive_label else 0
        )
        frame.to_csv(upload_path, index=False)

    ordinal_mappings = infer_ordinal_mappings(frame)
    config = DatasetConfig(
        name=dataset_id,
        display_name=f"Custom upload ({dataset_id})",
        target_column=payload.target_column,
        source_type="csv",
        csv_path=upload_path,
        ordinal_mappings=ordinal_mappings,
    )
    register_dataset(config)

    report = train_experiment(dataset_name=dataset_id)
    return {"status": "trained", "champion_model": report["champion_model"], "metrics": report}


@app.post("/train")
def train(dataset: str | None = None) -> dict:
    from credit_risk_platform.training.train import train_experiment

    report = train_experiment(dataset_name=dataset)
    return {"status": "trained", "champion_model": report["champion_model"], "metrics": report}


@app.get("/model-info")
def model_info() -> dict:
    settings = get_settings()
    try:
        bundle = load_model_bundle(settings.model_artifact_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "model_name": bundle["model_name"],
        "dataset_name": bundle.get("dataset_name"),
        "features": bundle["feature_names"],
        "target_definition": bundle["target_definition"],
        "saved_at": bundle.get("saved_at"),
    }


@app.get("/model-metrics")
def model_metrics() -> dict:
    path = get_settings().metrics_path
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="Metrics are unavailable. Run training first.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.get("/feature-importance")
def feature_importance() -> list[dict]:
    path = get_settings().feature_importance_path
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="Feature importance unavailable. Run training first."
        )
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/prediction-history")
def prediction_history(db: Session = Depends(get_db), limit: int = 100) -> list[dict]:
    rows = db.scalars(
        select(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit)
    ).all()
    return [
        {
            "id": row.id,
            "created_at": row.created_at.isoformat(),
            "model_name": row.model_name,
            "probability_default": row.probability_default,
            "decision": row.decision,
            "threshold": row.threshold,
            "features": row.features,
        }
        for row in rows
    ]
