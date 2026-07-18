from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from credit_risk_platform.config.settings import get_settings
from credit_risk_platform.database.models import PredictionHistory
from credit_risk_platform.models.registry import load_model_bundle


def score_applicant(features: dict[str, Any], db: Session | None = None) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_model_bundle(settings.model_artifact_path)
    model = bundle["model"]
    feature_names: list[str] = bundle["feature_names"]
    row = {name: features.get(name) for name in feature_names}
    frame = pd.DataFrame([row])
    pd_score = float(model.predict_proba(frame)[:, 1][0])
    threshold = float(bundle["metrics"]["model_comparison"][bundle["model_name"]]["threshold"])
    decision = "decline" if pd_score >= threshold else "approve"
    response = {
        "probability_default": pd_score,
        "decision": decision,
        "threshold": threshold,
        "model_name": bundle["model_name"],
    }
    if db is not None:
        db.add(
            PredictionHistory(
                model_name=bundle["model_name"],
                probability_default=pd_score,
                decision=decision,
                threshold=threshold,
                features=row,
            )
        )
        db.commit()
    return response
