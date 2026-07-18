import pandas as pd

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

    assert "credit_amount_per_month" in transformed.columns
    assert transformed.loc[0, "credit_amount_per_month"] == 100


def test_outlier_clipper_caps_extreme_values() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 1000]})
    clipper = OutlierClipper(lower=0.0, upper=0.75).fit(frame)

    transformed = clipper.transform(frame)

    assert transformed.max() < 1000


def test_woe_encoder_returns_numeric_columns() -> None:
    frame = pd.DataFrame({"grade": ["A", "A", "B", "C"]})
    target = pd.Series([0, 0, 1, 1])

    transformed = WeightOfEvidenceEncoder().fit(frame, target).transform(frame)

    assert transformed["grade"].dtype.kind in {"f", "i"}
