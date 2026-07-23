from pathlib import Path

import pandas as pd
import pytest
from credit_risk_platform.config.datasets import DatasetConfig, register_dataset
from credit_risk_platform.config.settings import Settings
from credit_risk_platform.training.train import train_experiment


def _synthetic_frame(rows: int = 40) -> pd.DataFrame:
    records = []
    for i in range(rows):
        records.append(
            {
                "income": 1000 + (i * 37 % 500),
                "loan_amount": 200 + (i * 53 % 800),
                "outcome": "bad" if i % 3 == 0 else "good",
            }
        )
    return pd.DataFrame(records)


def test_register_dataset_and_train_end_to_end(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    frame["outcome"] = frame["outcome"].apply(lambda value: 1 if value == "bad" else 0)
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    config = DatasetConfig(
        name="test_custom",
        display_name="Test custom dataset",
        target_column="outcome",
        source_type="csv",
        csv_path=csv_path,
    )
    register_dataset(config)

    settings = Settings(
        model_artifact_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        feature_importance_path=tmp_path / "feature_importance.csv",
        reports_dir=tmp_path / "reports",
    )
    report = train_experiment(dataset_name="test_custom", settings=settings)

    assert "champion_model" in report
    assert report["champion_model"] in report["model_comparison"]
    assert settings.model_artifact_path.exists()


def test_load_credit_dataset_rejects_non_numeric_multi_valued_target(tmp_path: Path) -> None:
    from credit_risk_platform.training.data import load_credit_dataset

    records = [{"income": 1000 + i, "segment": "abc"[i % 3]} for i in range(30)]
    frame = pd.DataFrame(records)
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    config = DatasetConfig(
        name="test_non_binary",
        display_name="Test non-binary dataset",
        target_column="segment",
        source_type="csv",
        csv_path=csv_path,
    )
    register_dataset(config)

    with pytest.raises(ValueError):
        load_credit_dataset("test_non_binary")
