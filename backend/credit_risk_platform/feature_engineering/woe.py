import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """Weight-of-evidence encoder for binary default modelling.

    The transformer is available for scorecard-style experiments. Tree and
    one-hot pipelines remain the default champion candidates because they keep
    the original German Credit categorical semantics transparent.
    """

    def __init__(self, smoothing: float = 0.5) -> None:
        self.smoothing = smoothing
        self.maps_: dict[str, dict[str, float]] = {}
        self.defaults_: dict[str, float] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "WeightOfEvidenceEncoder":
        target = pd.Series(y).astype(int)
        total_bad = target.sum() + self.smoothing
        total_good = len(target) - target.sum() + self.smoothing
        for column in x.columns:
            frame = pd.DataFrame({"feature": x[column].astype(str), "target": target})
            grouped = frame.groupby("feature", observed=False)["target"].agg(["sum", "count"])
            bad_dist = (grouped["sum"] + self.smoothing) / total_bad
            good_dist = (grouped["count"] - grouped["sum"] + self.smoothing) / total_good
            mapping = np.log(good_dist / bad_dist).to_dict()
            self.maps_[column] = {str(key): float(value) for key, value in mapping.items()}
            self.defaults_[column] = 0.0
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=x.index)
        for column in x.columns:
            mapping = self.maps_.get(column, {})
            transformed[column] = (
                x[column].astype(str).map(mapping).fillna(self.defaults_.get(column, 0.0))
            )
        return transformed
