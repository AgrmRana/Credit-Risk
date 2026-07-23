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


def test_supplied_threshold_is_used_verbatim() -> None:
    # An externally learned threshold must be applied as-is, not re-optimised on y_true, so that
    # test-set precision/recall/F1 reflect a cutoff chosen on separate (training) data.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.9])

    metrics = classification_metrics(y_true, y_score, threshold=0.5)

    assert metrics["threshold"] == 0.5
    # At 0.5 the two positives (0.6, 0.9) are caught and no negatives are, so recall == 1.0.
    assert metrics["recall"] == 1.0


def test_threshold_that_misses_all_positives_scores_zero_recall() -> None:
    # A deliberately bad supplied threshold is honoured (proving it isn't silently re-optimised).
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.9])

    metrics = classification_metrics(y_true, y_score, threshold=0.95)

    assert metrics["threshold"] == 0.95
    assert metrics["recall"] == 0.0
