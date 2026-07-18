from fastapi.testclient import TestClient

from credit_risk_platform.api.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_business_decision() -> None:
    payload = {
        "features": {
            "checking_status": "<0",
            "duration": 24,
            "credit_history": "existing paid",
            "purpose": "radio/tv",
            "credit_amount": 2500,
            "savings_status": "<100",
            "employment": "1<=X<4",
            "installment_commitment": 3,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 2,
            "property_magnitude": "real estate",
            "age": 35,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 1,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "none",
            "foreign_worker": "yes",
        }
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "probability_default" in body
    assert body["decision"] in {"Approve", "Manual Review", "Reject"}
    assert body["risk_band"] in {"Low", "Medium", "High", "Very High"}
    assert "prediction_confidence" in body
