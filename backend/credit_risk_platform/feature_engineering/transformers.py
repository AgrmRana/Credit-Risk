import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features by fitted quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: pd.Series | None = None
        self.upper_bounds_: pd.Series | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "OutlierClipper":
        values = np.asarray(x, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("OutlierClipper must be fitted before transform.")
        return np.clip(np.asarray(x, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.array([])
        return np.asarray(input_features, dtype=object)


class FeatureInteractionBuilder(BaseEstimator, TransformerMixin):
    """Create credit-domain numeric interaction features when inputs are present."""

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "FeatureInteractionBuilder":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        data = x.copy()
        if {"duration", "credit_amount"}.issubset(data.columns):
            data["credit_amount_per_month"] = data["credit_amount"] / np.maximum(
                data["duration"], 1
            )
        if {"installment_commitment", "duration"}.issubset(data.columns):
            data["installment_duration_interaction"] = (
                data["installment_commitment"] * data["duration"]
            )
        if {"age", "duration"}.issubset(data.columns):
            data["age_to_duration_ratio"] = data["age"] / np.maximum(data["duration"], 1)
        return data

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        features = list(input_features or [])
        if {"duration", "credit_amount"}.issubset(features):
            features.append("credit_amount_per_month")
        if {"installment_commitment", "duration"}.issubset(features):
            features.append("installment_duration_interaction")
        if {"age", "duration"}.issubset(features):
            features.append("age_to_duration_ratio")
        return np.asarray(features, dtype=object)
