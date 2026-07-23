import numpy as np
from credit_risk_platform.evaluation.metrics import classification_metrics, ks_statistic


def test_classification_metrics_are_computed() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.05, 0.3, 0.72, 0.91])

    metrics = classification_metrics(y_true, y_score)

    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks_statistic"] == 1.0
    assert 0.05 <= metrics["threshold"] <= 0.95


def test_ks_statistic_handles_ranked_scores() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.7, 0.2, 0.8, 0.9])

    assert ks_statistic(y_true, y_score) > 0.0
