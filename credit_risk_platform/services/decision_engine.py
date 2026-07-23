import json
from functools import lru_cache
from pathlib import Path
from typing import Any

THRESHOLD_PATH = Path(__file__).resolve().parents[1] / "config" / "decision_thresholds.json"


@lru_cache
def load_decision_thresholds() -> dict[str, Any]:
    return json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))


def assign_risk_band(probability_default: float) -> str:
    thresholds = load_decision_thresholds()
    for band in thresholds["risk_bands"]:
        if band["min_pd"] <= probability_default < band["max_pd"]:
            return str(band["name"])
    return str(thresholds["risk_bands"][-1]["name"])


def make_business_decision(probability_default: float) -> str:
    thresholds = load_decision_thresholds()
    if probability_default <= thresholds["approve_max_pd"]:
        return "Approve"
    if probability_default <= thresholds["manual_review_max_pd"]:
        return "Manual Review"
    return "Reject"


def prediction_confidence(probability_default: float) -> float:
    """Distance from the manual-review boundary as a simple operational confidence signal."""

    thresholds = load_decision_thresholds()
    lower = thresholds["approve_max_pd"]
    upper = thresholds["manual_review_max_pd"]
    if lower < probability_default <= upper:
        distance = min(probability_default - lower, upper - probability_default)
        return round(float(distance / ((upper - lower) / 2)), 4)
    return round(float(min(abs(probability_default - lower), abs(probability_default - upper))), 4)
