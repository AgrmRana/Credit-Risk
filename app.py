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

IMAGE_KEYS = ["roc_curve", "calibration_curve", "partial_dependence", "shap_summary"]


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
    frame: pd.DataFrame, target_column: str, positive_label: str | None, cv_folds: int
) -> dict[str, Any]:
    frame = frame.copy()
    if positive_label is not None:
        frame[target_column] = frame[target_column].apply(
            lambda value: 1 if str(value) == positive_label else 0
        )

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

    st.subheader(f"Champion model: {champion}")
    st.caption(
        f"Selected as the lowest {cv_folds}-fold cross-validated test MSE "
        "(out-of-fold Brier score) across all candidates."
    )

    leaderboard = (
        pd.DataFrame(comparison)
        .T[["cv_test_mse", "roc_auc", "pr_auc"]]
        .sort_values("cv_test_mse")
        .rename(columns={"cv_test_mse": f"cv_test_mse (k={cv_folds})"})
    )
    st.subheader("Leaderboard: Cross-Validated Test MSE")
    st.dataframe(
        leaderboard.style.highlight_min(subset=[f"cv_test_mse (k={cv_folds})"], color="#0f766e33"),
        use_container_width=True,
    )

    comparison_table = pd.DataFrame(comparison).T[
        ["roc_auc", "pr_auc", "ks_statistic", "gini", "precision", "recall", "f1", "threshold"]
    ]
    st.subheader("Full Model Comparison")
    st.dataframe(comparison_table, use_container_width=True)

    images = result["images"]
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
    feature_names: list[str] = bundle["feature_names"]

    with st.form("score_record_form"):
        values: dict[str, Any] = {}
        for name in feature_names:
            values[name] = st.text_input(name, key=f"score_{name}")
        submitted = st.form_submit_button("Score this record", type="primary")

    if submitted:
        features: dict[str, Any] = {}
        for name, raw_value in values.items():
            try:
                features[name] = float(raw_value)
            except ValueError:
                features[name] = raw_value
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

    binary_columns = [c for c in frame.columns if frame[c].dropna().nunique() == 2]
    if not binary_columns:
        st.error(
            "No column with exactly 2 unique values was found. This pipeline supports binary "
            "classification only, so the target column must have exactly two distinct values."
        )
        return

    target_column = st.selectbox("Target variable (what you want to predict)", binary_columns)
    uniques = sorted(frame[target_column].dropna().unique().tolist(), key=str)
    already_binary = {str(value) for value in uniques} <= {"0", "1"}
    positive_label = None
    if not already_binary:
        positive_label = st.radio(
            f"Which value of '{target_column}' represents the positive / default outcome?",
            [str(value) for value in uniques],
        )

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
                frame, target_column, positive_label, int(cv_folds)
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
