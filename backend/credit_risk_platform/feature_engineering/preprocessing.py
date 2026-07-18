import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_risk_platform.feature_engineering.transformers import (
    FeatureInteractionBuilder,
    OutlierClipper,
)

ORDINAL_CATEGORIES = {
    "checking_status": ["no checking", "<0", "0<=X<200", ">=200"],
    "savings_status": ["no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
}


def build_preprocessor(x_train: pd.DataFrame) -> Pipeline:
    ordinal_cols = [col for col in ORDINAL_CATEGORIES if col in x_train.columns]
    categorical_cols = [
        col
        for col in x_train.select_dtypes(include=["category", "object", "bool"]).columns
        if col not in ordinal_cols
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper()),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ORDINAL_CATEGORIES[col] for col in ordinal_cols],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = [
        ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
    ]
    if ordinal_cols:
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return Pipeline(
        steps=[
            ("interactions", FeatureInteractionBuilder()),
            ("columns", ColumnTransformer(transformers=transformers, remainder="drop")),
        ]
    )
