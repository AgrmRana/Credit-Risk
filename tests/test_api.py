import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_risk_platform.api.main import app
from credit_risk_platform.config.settings import get_settings


def _synthetic_csv(rows: int = 40) -> bytes:
    lines = ["income,loan_amount,outcome"]
    for i in range(rows):
        income = 1000 + (i * 37 % 500)
        loan_amount = 200 + (i * 53 % 800)
        outcome = "bad" if i % 3 == 0 else "good"
        lines.append(f"{income},{loan_amount},{outcome}")
    return ("\n".join(lines)).encode("utf-8")


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


def _cleanup_uploaded_dataset(dataset_id: str) -> None:
    upload_path = Path("data/raw/uploads") / f"{dataset_id}.csv"
    upload_path.unlink(missing_ok=True)


def test_custom_dataset_upload_detects_columns() -> None:
    csv_bytes = _synthetic_csv()

    with TestClient(app) as client:
        response = client.post(
            "/custom-datasets/upload",
            files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
        )

    body = response.json()
    try:
        assert response.status_code == 200
        assert body["row_count"] == 40
        column_names = {column["name"] for column in body["columns"]}
        assert column_names == {"income", "loan_amount", "outcome"}
        outcome_column = next(column for column in body["columns"] if column["name"] == "outcome")
        assert set(outcome_column["unique_values"]) == {"good", "bad"}
    finally:
        _cleanup_uploaded_dataset(body["dataset_id"])


def test_custom_dataset_train_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect the (otherwise single, global) trained-model artifact paths to a
    # temp dir for this test only, so it doesn't clobber the real deployed
    # model that other tests (and the running app) rely on.
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("METRICS_PATH", str(tmp_path / "metrics.json"))
    monkeypatch.setenv("FEATURE_IMPORTANCE_PATH", str(tmp_path / "feature_importance.csv"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()

    csv_bytes = _synthetic_csv()
    dataset_id = None
    try:
        with TestClient(app) as client:
            upload_response = client.post(
                "/custom-datasets/upload",
                files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
            )
            dataset_id = upload_response.json()["dataset_id"]

            train_response = client.post(
                f"/custom-datasets/{dataset_id}/train",
                json={"target_column": "outcome", "positive_label": "bad"},
            )

        assert train_response.status_code == 200
        body = train_response.json()
        assert body["status"] == "trained"
        assert "champion_model" in body
    finally:
        get_settings.cache_clear()
        if dataset_id:
            _cleanup_uploaded_dataset(dataset_id)


def test_custom_dataset_train_rejects_non_binary_target() -> None:
    lines = ["income,segment"]
    for i in range(30):
        lines.append(f"{1000 + i},{'abc'[i % 3]}")
    csv_bytes = ("\n".join(lines)).encode("utf-8")

    dataset_id = None
    try:
        with TestClient(app) as client:
            upload_response = client.post(
                "/custom-datasets/upload",
                files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")},
            )
            dataset_id = upload_response.json()["dataset_id"]

            train_response = client.post(
                f"/custom-datasets/{dataset_id}/train",
                json={"target_column": "segment"},
            )

        assert train_response.status_code == 400
    finally:
        if dataset_id:
            _cleanup_uploaded_dataset(dataset_id)
