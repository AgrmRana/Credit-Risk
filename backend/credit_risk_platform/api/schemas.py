from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ..., description="Applicant feature payload keyed by model feature name."
    )


class PredictionResponse(BaseModel):
    probability_default: float
    decision: str
    risk_band: str
    prediction_confidence: float
    threshold: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class CustomDatasetTrainRequest(BaseModel):
    target_column: str
    positive_label: str | None = None
