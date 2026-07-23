"""Local credit-risk analysis tool: upload a CSV, pick a target, run the pipeline, see results.

Nothing is written to disk beyond a temporary directory that is cleaned up automatically.
Nothing persists between browser sessions or app restarts.
"""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from credit_risk_platform.config.datasets import DatasetConfig, register_dataset
from credit_risk_platform.config.settings import Settings
from credit_risk_platform.feature_engineering.schema import infer_ordinal_mappings, profile_schema
from credit_risk_platform.models.registry import load_model_bundle
from credit_risk_platform.services.prediction import score_applicant
from credit_risk_platform.training.train import train_experiment

IMAGE_KEYS = [
    "roc_curve",
    "calibration_curve",
    "partial_dependence",
    "shap_summary",
    "confusion_matrix",
]


def _detect_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    ordinal_mappings = infer_ordinal_mappings(frame)
    profile = profile_schema(frame, ordinal_mappings)
    type_by_column: dict[str, str] = {}
    for column in profile.date_columns:
        type_by_column[column] = "date"
    for column in profile.boolean_columns:
        type_by_column[column] = "boolean"
    for column in profile.ordinal_columns:
        type_by_column[column] = "ordinal"
    for column in profile.numeric_columns:
        type_by_column[column] = "numeric"
    for column in profile.categorical_columns:
        type_by_column[column] = "categorical"
    for column in profile.dropped_columns:
        type_by_column[column] = "dropped"

    detected = pd.DataFrame(
        {
            "column": frame.columns,
            "detected_type": [type_by_column.get(c, "categorical") for c in frame.columns],
            "unique_values": [frame[c].nunique(dropna=True) for c in frame.columns],
        }
    )
    return detected, ordinal_mappings


def run_pipeline(
    frame: pd.DataFrame,
    target_column: str,
    class_mapping: dict[str, int] | None,
    cv_folds: int,
) -> dict[str, Any]:
    frame = frame.copy()
    class_names: dict[int, str] | None = None
    if class_mapping is not None:
        frame[target_column] = frame[target_column].apply(lambda value: class_mapping[str(value)])
        class_names = {code: label for label, code in class_mapping.items()}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "data.csv"
        frame.to_csv(csv_path, index=False)

        dataset_id = "session"
        config = DatasetConfig(
            name=dataset_id,
            display_name="Uploaded dataset",
            target_column=target_column,
            source_type="csv",
            csv_path=csv_path,
            ordinal_mappings=infer_ordinal_mappings(frame),
        )
        register_dataset(config)

        settings = Settings(
            model_artifact_path=tmp_path / "model.joblib",
            metrics_path=tmp_path / "metrics.json",
            feature_importance_path=tmp_path / "feature_importance.csv",
            reports_dir=tmp_path / "reports",
        )
        metrics_report = train_experiment(
            dataset_name=dataset_id,
            settings=settings,
            cv_folds=cv_folds,
            selection_metric="cv_test_mse",
            class_names=class_names,
        )
        bundle = load_model_bundle(settings.model_artifact_path)

        artifact_paths = {
            **metrics_report.get("validation_artifacts", {}),
            **metrics_report.get("explainability_artifacts", {}),
        }
        images = {
            key: Path(path).read_bytes()
            for key, path in artifact_paths.items()
            if key in IMAGE_KEYS and Path(path).exists()
        }
        lift_gain = None
        if "lift_gain" in artifact_paths and Path(artifact_paths["lift_gain"]).exists():
            lift_gain = pd.read_csv(artifact_paths["lift_gain"])

        feature_report_paths = metrics_report.get("feature_report_artifacts", {})
        feature_report_markdown = None
        if "markdown" in feature_report_paths and Path(feature_report_paths["markdown"]).exists():
            feature_report_markdown = Path(feature_report_paths["markdown"]).read_text(
                encoding="utf-8"
            )

    return {
        "metrics": metrics_report,
        "bundle": bundle,
        "images": images,
        "lift_gain": lift_gain,
        "feature_report_markdown": feature_report_markdown,
    }


def render_results(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    comparison = metrics["model_comparison"]
    champion = metrics["champion_model"]
    cv_folds = metrics.get("cv_folds")
    num_classes = metrics.get("num_classes", 2)
    class_names = metrics.get("class_names") or {}

    st.subheader(f"Champion model: {champion}")
    st.caption(
        f"Selected as the lowest {cv_folds}-fold cross-validated test MSE "
        "(out-of-fold Brier score) across all candidates."
    )
    if num_classes > 2:
        class_list = ", ".join(f"'{label}' → {code}" for code, label in sorted(class_names.items()))
        st.caption(f"Multi-class target with {num_classes} classes: {class_list}")

    leaderboard_columns = ["cv_test_mse", "roc_auc"] + (
        ["accuracy"] if num_classes > 2 else ["pr_auc"]
    )
    leaderboard = (
        pd.DataFrame(comparison)
        .T[leaderboard_columns]
        .sort_values("cv_test_mse")
        .rename(columns={"cv_test_mse": f"cv_test_mse (k={cv_folds})"})
    )
    st.subheader("Leaderboard: Cross-Validated Test MSE")
    st.dataframe(
        leaderboard.style.highlight_min(subset=[f"cv_test_mse (k={cv_folds})"], color="#0f766e33"),
        use_container_width=True,
    )

    if num_classes > 2:
        comparison_columns = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    else:
        comparison_columns = [
            "roc_auc",
            "pr_auc",
            "ks_statistic",
            "gini",
            "precision",
            "recall",
            "f1",
            "threshold",
        ]
    comparison_table = pd.DataFrame(comparison).T[comparison_columns]
    st.subheader("Full Model Comparison")
    st.dataframe(comparison_table, use_container_width=True)

    images = result["images"]
    if num_classes > 2:
        if "confusion_matrix" in images:
            st.subheader("Confusion Matrix")
            st.image(images["confusion_matrix"])
    else:
        cols = st.columns(2)
        for index, key in enumerate(["roc_curve", "calibration_curve"]):
            if key in images:
                cols[index % 2].image(images[key], caption=key.replace("_", " ").title())

        if result["lift_gain"] is not None:
            st.subheader("Lift / Gain")
            st.dataframe(result["lift_gain"], use_container_width=True)

    explain_cols = st.columns(2)
    for index, key in enumerate(["shap_summary", "partial_dependence"]):
        if key in images:
            explain_cols[index % 2].image(images[key], caption=key.replace("_", " ").title())

    summary = metrics.get("feature_engineering_summary", {})
    st.subheader("Feature Engineering Summary")
    st.write(
        {
            "original_features": summary.get("original_features", []),
            "derived_features": summary.get("derived_features", []),
            "dropped_features": summary.get("dropped_features", []),
        }
    )

    if result["feature_report_markdown"]:
        with st.expander("Full feature engineering report"):
            st.markdown(result["feature_report_markdown"])


def render_scoring(result: dict[str, Any]) -> None:
    bundle = result["bundle"]
    metrics = result["metrics"]
    num_classes = metrics.get("num_classes", 2)
    class_names = metrics.get("class_names") or {}
    feature_names: list[str] = bundle["feature_names"]

    with st.form("score_record_form"):
        values: dict[str, Any] = {}
        for name in feature_names:
            values[name] = st.text_input(name, key=f"score_{name}")
        submitted = st.form_submit_button("Score this record", type="primary")

    if not submitted:
        return

    features: dict[str, Any] = {}
    for name, raw_value in values.items():
        try:
            features[name] = float(raw_value)
        except ValueError:
            features[name] = raw_value

    if num_classes == 2:
        response = score_applicant(features, model_bundle=bundle)
        badge = {
            "Approve": st.success,
            "Reject": st.error,
            "Manual Review": st.warning,
        }.get(response["decision"], st.info)
        badge(
            f"{response['decision']} — PD {response['probability_default']:.2%}, "
            f"risk band {response['risk_band']}, "
            f"confidence {response['prediction_confidence']:.2%}"
        )
        return

    row = {name: features.get(name) for name in feature_names}
    frame = pd.DataFrame([row])
    probabilities = bundle["model"].predict_proba(frame)[0]
    predicted_index = int(probabilities.argmax())
    predicted_label = class_names.get(predicted_index, str(predicted_index))
    st.success(
        f"Predicted class: {predicted_label} ({probabilities[predicted_index]:.2%} confidence)"
    )
    labels = [class_names.get(index, str(index)) for index in range(len(probabilities))]
    st.bar_chart(pd.DataFrame({"probability": probabilities}, index=labels))


def main() -> None:
    st.set_page_config(page_title="Credit Risk Decision Platform", layout="wide")
    st.title("Credit Risk Decision Platform")
    st.caption(
        "Upload a CSV, choose the target variable, and run the full model-comparison pipeline. "
        "Everything runs in memory — nothing is saved once the app closes."
    )

    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()

    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is None:
        st.session_state.pop("result", None)
        return

    frame = pd.read_csv(uploaded)
    detected, _ = _detect_columns(frame)
    st.subheader("Detected Columns")
    st.dataframe(detected, use_container_width=True)

    classification_columns = [c for c in frame.columns if 2 <= frame[c].dropna().nunique() <= 20]
    if not classification_columns:
        st.error(
            "No column with between 2 and 20 unique values was found. The target needs to be "
            "categorical (binary or multi-class) with a reasonably small number of distinct "
            "values — a continuous numeric column or an ID-like column with many unique values "
            "can't be used as a classification target."
        )
        return

    target_column = st.selectbox(
        "Target variable (what you want to predict)", classification_columns
    )
    uniques = sorted(frame[target_column].dropna().unique().tolist(), key=str)
    already_binary = len(uniques) == 2 and {str(value) for value in uniques} <= {"0", "1"}

    class_mapping: dict[str, int] | None = None
    if already_binary:
        pass
    elif len(uniques) == 2:
        positive_label = st.radio(
            f"Which value of '{target_column}' represents the positive / default outcome?",
            [str(value) for value in uniques],
        )
        other_value = next(str(value) for value in uniques if str(value) != positive_label)
        class_mapping = {other_value: 0, positive_label: 1}
    else:
        class_mapping = {str(value): index for index, value in enumerate(uniques)}
        class_list = ", ".join(f"'{value}' → {index}" for value, index in class_mapping.items())
        st.info(f"Multi-class target detected ({len(uniques)} classes): {class_list}")

    min_class_count = int(frame[target_column].dropna().value_counts().min())
    max_folds = max(2, min_class_count)
    cv_folds = st.number_input(
        "Number of cross-validation folds (k)",
        min_value=2,
        max_value=max_folds,
        value=min(5, max_folds),
        step=1,
        help=(
            "Each candidate model is evaluated with k-fold cross validation; the model with "
            "the lowest cross-validated test MSE (out-of-fold Brier score) becomes the champion."
        ),
    )

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Training and comparing candidate models..."):
            st.session_state["result"] = run_pipeline(
                frame, target_column, class_mapping, int(cv_folds)
            )

    result = st.session_state.get("result")
    if result:
        tab_results, tab_score = st.tabs(["Results", "Score a Record"])
        with tab_results:
            render_results(result)
        with tab_score:
            render_scoring(result)


if __name__ == "__main__":
    main()
