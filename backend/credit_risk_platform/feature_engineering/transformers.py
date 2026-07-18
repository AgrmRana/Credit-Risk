import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk_platform.feature_engineering.schema import (
    infer_semantic_column,
    infer_semantic_columns,
    matches_any,
    normalize_name,
)


class SelectiveOutlierClipper(BaseEstimator, TransformerMixin):
    """Clip only heavy-tailed numeric columns fitted from the training data."""

    def __init__(
        self, lower: float = 0.01, upper: float = 0.99, skew_threshold: float = 3.0
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.skew_threshold = skew_threshold
        self.clip_indices_: list[int] = []
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | None = None
    ) -> "SelectiveOutlierClipper":
        values = np.asarray(x, dtype=float)
        skew = pd.DataFrame(values).skew(numeric_only=True).abs().fillna(0)
        self.clip_indices_ = skew[skew >= self.skew_threshold].index.tolist()
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("SelectiveOutlierClipper must be fitted before transform.")
        values = np.asarray(x, dtype=float).copy()
        if self.clip_indices_:
            values[:, self.clip_indices_] = np.clip(
                values[:, self.clip_indices_],
                self.lower_bounds_[self.clip_indices_],
                self.upper_bounds_[self.clip_indices_],
            )
        return values

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


class DynamicCreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create credit-risk features from detected semantics only when source columns exist."""

    def __init__(self) -> None:
        self.derived_features_: list[str] = []
        self.date_columns_: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "DynamicCreditFeatureBuilder":
        self.derived_features_ = self._derive(x, collect_only=True)[1]
        self.date_columns_ = [
            column
            for column in x.columns
            if pd.api.types.is_datetime64_any_dtype(x[column])
            or (
                pd.api.types.is_object_dtype(x[column])
                and pd.to_datetime(
                    x[column].dropna().astype(str).head(100),
                    errors="coerce",
                    format="mixed",
                )
                .notna()
                .mean()
                >= 0.8
            )
        ]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._derive(x, collect_only=False)[0]

    def _derive(self, x: pd.DataFrame, collect_only: bool) -> tuple[pd.DataFrame, list[str]]:
        data = x.copy()
        derived: list[str] = []
        numeric_columns = list(data.select_dtypes(include="number").columns)
        all_columns = list(data.columns)

        loan_amount = infer_semantic_column(
            numeric_columns,
            [r"loan.*amount", r"credit.*amount", r"amt.*credit", r"amount"],
            [r"income", r"annuity", r"limit", r"balance"],
        )
        duration = infer_semantic_column(
            numeric_columns,
            [r"duration", r"term", r"tenor", r"months"],
            [r"employment", r"since"],
        )
        income = infer_semantic_column(
            numeric_columns,
            [r"income", r"salary", r"earnings"],
        )
        savings = infer_semantic_column(numeric_columns, [r"saving", r"asset", r"deposit"])
        age = infer_semantic_column(numeric_columns, [r"age", r"days_birth"])
        employment_duration = infer_semantic_column(
            numeric_columns,
            [r"employment.*duration", r"employed", r"employment", r"years.*job", r"days_employed"],
        )
        existing_loans = infer_semantic_column(
            numeric_columns,
            [r"existing.*credit", r"existing.*loan", r"num.*loan", r"cnt.*credit", r"open.*credit"],
        )
        revolving_balance = infer_semantic_column(
            numeric_columns,
            [r"revolving.*balance", r"revol.*bal", r"balance"],
            [r"limit", r"income"],
        )
        credit_limit = infer_semantic_column(
            numeric_columns,
            [r"credit.*limit", r"limit"],
            [r"amount"],
        )
        delinquency_columns = infer_semantic_columns(
            numeric_columns,
            [r"delinq", r"dpd", r"past_due", r"late", r"missed", r"times.*worse"],
        )
        payment_columns = infer_semantic_columns(numeric_columns, [r"payment", r"installment"])

        def add_feature(name: str, values: pd.Series) -> None:
            if collect_only:
                derived.append(name)
            else:
                data[name] = values.replace([np.inf, -np.inf], np.nan)
                derived.append(name)

        if loan_amount and duration:
            add_feature(
                f"{normalize_name(loan_amount)}_per_{normalize_name(duration)}",
                pd.to_numeric(data[loan_amount], errors="coerce")
                / np.maximum(pd.to_numeric(data[duration], errors="coerce").abs(), 1),
            )
        if income and loan_amount:
            add_feature(
                "debt_to_income_ratio",
                pd.to_numeric(data[loan_amount], errors="coerce")
                / np.maximum(pd.to_numeric(data[income], errors="coerce").abs(), 1),
            )
        if savings and loan_amount:
            add_feature(
                "savings_to_loan_ratio",
                pd.to_numeric(data[savings], errors="coerce")
                / np.maximum(pd.to_numeric(data[loan_amount], errors="coerce").abs(), 1),
            )
        if age:
            age_values = pd.to_numeric(data[age], errors="coerce")
            if matches_any(age, [r"days_birth"]):
                age_values = age_values.abs() / 365.25
            add_feature("age_squared", age_values**2)
            add_feature(
                "age_band", pd.cut(age_values, bins=[0, 25, 35, 45, 55, 65, 120], labels=False)
            )
        if employment_duration:
            employment_values = pd.to_numeric(data[employment_duration], errors="coerce").abs()
            if matches_any(employment_duration, [r"days"]):
                employment_values = employment_values / 365.25
            add_feature("employment_stability_score", np.log1p(employment_values))
        if existing_loans and loan_amount:
            add_feature(
                "credit_exposure_score",
                pd.to_numeric(data[existing_loans], errors="coerce")
                * pd.to_numeric(data[loan_amount], errors="coerce"),
            )
        if revolving_balance and credit_limit:
            add_feature(
                "credit_utilization_ratio",
                pd.to_numeric(data[revolving_balance], errors="coerce")
                / np.maximum(pd.to_numeric(data[credit_limit], errors="coerce").abs(), 1),
            )
        if delinquency_columns:
            delinquency = data[delinquency_columns].apply(pd.to_numeric, errors="coerce")
            add_feature("delinquency_count", delinquency.gt(0).sum(axis=1))
            if payment_columns:
                payments = data[payment_columns].apply(pd.to_numeric, errors="coerce")
                add_feature(
                    "missed_payment_ratio",
                    delinquency.gt(0).sum(axis=1) / np.maximum(payments.notna().sum(axis=1), 1),
                )

        for column in all_columns:
            if not (
                pd.api.types.is_datetime64_any_dtype(data[column])
                or (
                    pd.api.types.is_object_dtype(data[column])
                    and pd.to_datetime(
                        data[column].dropna().astype(str).head(100),
                        errors="coerce",
                        format="mixed",
                    )
                    .notna()
                    .mean()
                    >= 0.8
                )
            ):
                continue
            parsed = pd.to_datetime(data[column], errors="coerce", format="mixed")
            prefix = normalize_name(column)
            add_feature(f"{prefix}_month", parsed.dt.month)
            add_feature(f"{prefix}_quarter", parsed.dt.quarter)
            add_feature(
                f"{prefix}_age_days", (pd.Timestamp.utcnow().tz_localize(None) - parsed).dt.days
            )

        return data, derived

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        features = [] if input_features is None else list(input_features)
        return np.asarray(features + self.derived_features_, dtype=object)


OutlierClipper = SelectiveOutlierClipper
FeatureInteractionBuilder = DynamicCreditFeatureBuilder
