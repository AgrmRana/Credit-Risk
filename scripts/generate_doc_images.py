import json
import shutil
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from credit_risk_platform.evaluation.metrics import lift_gain_table
from credit_risk_platform.training.data import make_train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS_IMAGES = ROOT / "docs" / "images"
ARTIFACTS = ROOT / "artifacts"

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main() -> None:
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    bundle = joblib.load(ARTIFACTS / "model.joblib")
    dataset_name = bundle.get("dataset_name", metrics.get("dataset_name", "german"))
    split = make_train_test_split(dataset_name=dataset_name)
    model = bundle["model"]
    y_true = split.y_test.to_numpy()
    y_score = model.predict_proba(split.x_test)[:, 1]
    champion_metrics = metrics["model_comparison"][metrics["champion_model"]]
    threshold = champion_metrics["threshold"]
    y_pred = y_score >= threshold

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(7, 4.5))
    plt.plot(fpr, tpr, color="#0f766e", linewidth=2, label=f"AUC {champion_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color="#8a94a6", linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_current_figure(DOCS_IMAGES / "roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(7, 4.5))
    plt.plot(recall, precision, color="#2563eb", linewidth=2)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    save_current_figure(DOCS_IMAGES / "precision_recall_curve.png")

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 4.5))
    plt.plot(prob_pred, prob_true, marker="o", color="#7c3aed")
    plt.plot([0, 1], [0, 1], color="#8a94a6", linestyle="--")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted PD")
    plt.ylabel("Observed Default Rate")
    save_current_figure(DOCS_IMAGES / "calibration_curve.png")

    lift_gain = lift_gain_table(y_true, y_score)
    plt.figure(figsize=(7, 4.5))
    plt.bar(lift_gain["bucket"].astype(str), lift_gain["lift"], color="#0f766e")
    plt.title("Lift Chart")
    plt.xlabel("Score Decile")
    plt.ylabel("Lift")
    save_current_figure(DOCS_IMAGES / "lift_chart.png")

    plt.figure(figsize=(7, 4.5))
    plt.plot(lift_gain["bucket"], lift_gain["gain"], marker="o", color="#2563eb")
    plt.title("Gain Chart")
    plt.xlabel("Score Decile")
    plt.ylabel("Cumulative Captured Defaults")
    save_current_figure(DOCS_IMAGES / "gain_chart.png")

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Pred Non-default", "Pred Default"], rotation=15)
    plt.yticks([0, 1], ["Actual Non-default", "Actual Default"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="#111827")
    plt.colorbar(fraction=0.046, pad=0.04)
    save_current_figure(DOCS_IMAGES / "confusion_matrix.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(y_score[y_true == 0], bins=20, alpha=0.7, label="Non-default", color="#2563eb")
    plt.hist(y_score[y_true == 1], bins=20, alpha=0.7, label="Default", color="#dc2626")
    plt.axvline(threshold, color="#111827", linestyle="--", label=f"Threshold {threshold:.2f}")
    plt.title("Prediction Distribution")
    plt.xlabel("Predicted PD")
    plt.ylabel("Applicants")
    plt.legend()
    save_current_figure(DOCS_IMAGES / "prediction_distribution.png")

    importance = pd.read_csv(ARTIFACTS / "feature_importance.csv").head(12)
    plt.figure(figsize=(8, 5.5))
    plt.barh(importance["feature"][::-1], importance["importance_mean"][::-1], color="#0f766e")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean ROC AUC Decrease")
    save_current_figure(DOCS_IMAGES / "feature_importance.png")

    residuals = y_true - y_score
    plt.figure(figsize=(7, 4.5))
    plt.scatter(y_score, residuals, alpha=0.75, color="#475467")
    plt.axhline(0, color="#111827", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted PD")
    plt.ylabel("Actual Default - Predicted PD")
    save_current_figure(DOCS_IMAGES / "residual_plot.png")

    shap_source = ARTIFACTS / "reports" / "shap_summary.png"
    if shap_source.exists():
        shutil.copyfile(shap_source, DOCS_IMAGES / "shap_summary.png")

    print(f"Wrote documentation images to {DOCS_IMAGES}")


if __name__ == "__main__":
    main()
