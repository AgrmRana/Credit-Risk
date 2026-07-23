import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def optimize_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, y_score >= threshold, zero_division=0) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def lift_gain_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y_true, "score": y_score}).sort_values("score", ascending=False)
    df["bucket"] = pd.qcut(np.arange(len(df)), bins, labels=False) + 1
    base_rate = df["actual"].mean()
    table = (
        df.groupby("bucket", observed=False)
        .agg(applicants=("actual", "size"), defaults=("actual", "sum"), avg_pd=("score", "mean"))
        .reset_index()
    )
    table["cumulative_defaults"] = table["defaults"].cumsum()
    table["gain"] = table["cumulative_defaults"] / max(table["defaults"].sum(), 1)
    table["lift"] = table["avg_pd"] / base_rate if base_rate else 0
    return table


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, num_classes: int = 2
) -> dict[str, Any]:
    """Compute metrics for a binary or multi-class classifier.

    ``y_score`` is the full ``predict_proba`` output. For binary targets (``num_classes == 2``)
    a 1-D array of just the positive-class probability is also accepted for convenience.
    Multi-class metrics use macro-averaging and drop the binary-only concepts (KS statistic,
    Gini, PR AUC, a single decision threshold) that have no standard multi-class equivalent.
    """
    if num_classes == 2:
        positive_score = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
        threshold, best_f1 = optimize_threshold(y_true, positive_score)
        y_pred = positive_score >= threshold
        precision, recall, _ = precision_recall_curve(y_true, positive_score)
        return {
            "roc_auc": float(roc_auc_score(y_true, positive_score)),
            "pr_auc": float(average_precision_score(y_true, positive_score)),
            "ks_statistic": ks_statistic(y_true, positive_score),
            "gini": float(2 * roc_auc_score(y_true, positive_score) - 1),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(best_f1),
            "threshold": threshold,
            "pr_curve_auc": float(auc(recall, precision)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    y_pred = np.argmax(y_score, axis=1)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def write_validation_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_dir: Path,
    prefix: str = "champion",
    num_classes: int = 2,
    class_names: dict[int, str] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if num_classes > 2:
        y_pred = np.argmax(y_score, axis=1)
        matrix = confusion_matrix(y_true, y_pred)
        labels = [(class_names or {}).get(index, str(index)) for index in range(num_classes)]
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, matrix[row, col], ha="center", va="center", color="#111827")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / f"{prefix}_confusion_matrix.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["confusion_matrix"] = str(path)
        return paths

    positive_score = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
    fpr, tpr, _ = roc_curve(y_true, positive_score)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    path = output_dir / f"{prefix}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["roc_curve"] = str(path)

    prob_true, prob_pred = calibration_curve(y_true, positive_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(title="Calibration Curve", xlabel="Mean Predicted PD", ylabel="Observed Default Rate")
    path = output_dir / f"{prefix}_calibration_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["calibration_curve"] = str(path)

    table = lift_gain_table(y_true, positive_score)
    table.to_csv(output_dir / f"{prefix}_lift_gain.csv", index=False)
    paths["lift_gain"] = str(output_dir / f"{prefix}_lift_gain.csv")
    return paths


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
