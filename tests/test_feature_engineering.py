import pandas as pd
from credit_risk_platform.feature_engineering.preprocessing import build_preprocessor
from credit_risk_platform.feature_engineering.transformers import (
    FeatureInteractionBuilder,
    OutlierClipper,
)
from credit_risk_platform.feature_engineering.woe import WeightOfEvidenceEncoder


def test_feature_interactions_are_created() -> None:
    frame = pd.DataFrame(
        {
            "duration": [10, 20],
            "credit_amount": [1000, 5000],
            "installment_commitment": [2, 4],
            "age": [30, 50],
        }
    )

    transformed = FeatureInteractionBuilder().fit_transform(frame)

    assert "credit_amount_per_duration" in transformed.columns
    assert "age_squared" in transformed.columns
    assert transformed.loc[0, "credit_amount_per_duration"] == 100


def test_outlier_clipper_caps_extreme_values() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 1000]})
    clipper = OutlierClipper(lower=0.0, upper=0.75, skew_threshold=0.0).fit(frame)

    transformed = clipper.transform(frame)

    assert transformed.max() < 1000


def test_woe_encoder_returns_numeric_columns() -> None:
    frame = pd.DataFrame({"grade": ["A", "A", "B", "C"]})
    target = pd.Series([0, 0, 1, 1])

    transformed = WeightOfEvidenceEncoder().fit(frame, target).transform(frame)

    assert transformed["grade"].dtype.kind in {"f", "i"}


def test_preprocessor_adapts_to_mixed_schema() -> None:
    frame = pd.DataFrame(
        {
            "loan_amount": [1000, 2000, 5000, 7000],
            "income": [50000, 60000, 80000, 90000],
            "risk_grade": ["low", "medium", "high", "medium"],
            "product_type": ["card", "loan", "loan", "card"],
            "application_date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "constant": [1, 1, 1, 1],
        }
    )

    preprocessor, report = build_preprocessor(frame)
    transformed = preprocessor.fit_transform(frame)

    assert transformed.shape[0] == len(frame)
    assert "debt_to_income_ratio" in report.derived_features
    assert "constant" in report.dropped_features
    assert "risk_grade" in report.ordinal_columns
    assert "product_type" in report.categorical_columns
    assert "application_date" in report.date_columns
