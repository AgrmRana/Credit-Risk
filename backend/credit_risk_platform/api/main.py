import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from credit_risk_platform.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from credit_risk_platform.config.settings import get_settings
from credit_risk_platform.database.init_db import init_db
from credit_risk_platform.database.models import PredictionHistory
from credit_risk_platform.database.session import get_db
from credit_risk_platform.models.registry import load_model_bundle
from credit_risk_platform.services.prediction import score_applicant


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


@app.post("/train")
def train() -> dict:
    from credit_risk_platform.training.train import train_experiment

    report = train_experiment()
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
