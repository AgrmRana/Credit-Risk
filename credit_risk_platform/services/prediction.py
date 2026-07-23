from typing import Any

import pandas as pd

from credit_risk_platform.config.settings import get_settings
from credit_risk_platform.models.registry import load_model_bundle
from credit_risk_platform.services.decision_engine import (
    assign_risk_band,
    make_business_decision,
    prediction_confidence,
)


def score_applicant(
    features: dict[str, Any], model_bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    bundle = model_bundle or load_model_bundle(get_settings().model_artifact_path)
    model = bundle["model"]
    feature_names: list[str] = bundle["feature_names"]
    row = {name: features.get(name) for name in feature_names}
    frame = pd.DataFrame([row])
    pd_score = float(model.predict_proba(frame)[:, 1][0])
    threshold = float(bundle["metrics"]["model_comparison"][bundle["model_name"]]["threshold"])
    return {
        "probability_default": pd_score,
        "decision": make_business_decision(pd_score),
        "risk_band": assign_risk_band(pd_score),
        "prediction_confidence": prediction_confidence(pd_score),
        "threshold": threshold,
        "model_name": bundle["model_name"],
    }
