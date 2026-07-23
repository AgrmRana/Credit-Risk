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


def test_cv_folds_and_mse_selection(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    frame["outcome"] = frame["outcome"].apply(lambda value: 1 if value == "bad" else 0)
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    config = DatasetConfig(
        name="test_cv_folds",
        display_name="Test CV folds dataset",
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
    report = train_experiment(
        dataset_name="test_cv_folds",
        settings=settings,
        cv_folds=3,
        selection_metric="cv_test_mse",
    )

    comparison = report["model_comparison"]
    assert report["cv_folds"] == 3
    assert report["selection_metric"] == "cv_test_mse"
    for metrics in comparison.values():
        assert metrics["cv_folds"] == 3
        assert "cv_test_mse" in metrics

    champion_mse = comparison[report["champion_model"]]["cv_test_mse"]
    assert champion_mse == min(metrics["cv_test_mse"] for metrics in comparison.values())


def test_default_champion_selection_uses_cross_validated_metric(tmp_path: Path) -> None:
    # The champion must be chosen on a cross-validated (train-only) metric so the held-out test
    # set is never used to pick among candidates — otherwise the reported test metrics are
    # selection-biased. The default selection_metric must therefore be a cv_* quantity.
    frame = _synthetic_frame()
    frame["outcome"] = frame["outcome"].apply(lambda value: 1 if value == "bad" else 0)
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    config = DatasetConfig(
        name="test_default_selection",
        display_name="Test default selection dataset",
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
    report = train_experiment(dataset_name="test_default_selection", settings=settings)

    selection_metric = report["selection_metric"]
    assert selection_metric.startswith("cv_")
    comparison = report["model_comparison"]
    champion_score = comparison[report["champion_model"]][selection_metric]
    assert champion_score == max(metrics[selection_metric] for metrics in comparison.values())


def test_multiclass_target_end_to_end(tmp_path: Path) -> None:
    labels = ["A", "B", "C"]
    records = []
    for i in range(60):
        label = labels[i % 3]
        base = {"A": 1000, "B": 1500, "C": 2000}[label]
        records.append(
            {
                "income": base + (i * 17 % 200),
                "loan_amount": 300 + (i * 29 % 400),
                "outcome": label,
            }
        )
    frame = pd.DataFrame(records)
    class_mapping = {value: index for index, value in enumerate(labels)}
    frame["outcome"] = frame["outcome"].map(class_mapping)
    csv_path = tmp_path / "data.csv"
    frame.to_csv(csv_path, index=False)

    config = DatasetConfig(
        name="test_multiclass",
        display_name="Test multiclass dataset",
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
    class_names = {code: label for label, code in class_mapping.items()}
    report = train_experiment(
        dataset_name="test_multiclass",
        settings=settings,
        cv_folds=3,
        selection_metric="cv_test_mse",
        class_names=class_names,
    )

    assert report["num_classes"] == 3
    assert report["class_names"] == class_names
    champion_metrics = report["model_comparison"][report["champion_model"]]
    assert "accuracy" in champion_metrics
    assert "ks_statistic" not in champion_metrics
    matrix = champion_metrics["confusion_matrix"]
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert "confusion_matrix" in report["validation_artifacts"]
    assert Path(report["validation_artifacts"]["confusion_matrix"]).exists()


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
