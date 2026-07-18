from credit_risk_platform.services.decision_engine import (
    assign_risk_band,
    make_business_decision,
    prediction_confidence,
)


def test_business_decision_thresholds() -> None:
    assert make_business_decision(0.2) == "Approve"
    assert make_business_decision(0.45) == "Manual Review"
    assert make_business_decision(0.7) == "Reject"


def test_risk_bands_and_confidence_are_returned() -> None:
    assert assign_risk_band(0.1) == "Low"
    assert assign_risk_band(0.3) == "Medium"
    assert assign_risk_band(0.45) == "High"
    assert assign_risk_band(0.8) == "Very High"
    assert prediction_confidence(0.45) >= 0
