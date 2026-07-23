import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_risk_platform.feature_engineering.schema import (
    FeatureEngineeringReport,
    infer_ordinal_mappings,
    profile_schema,
)
from credit_risk_platform.feature_engineering.transformers import (
    DynamicCreditFeatureBuilder,
    SelectiveOutlierClipper,
)


def build_preprocessor(
    x_train: pd.DataFrame,
    ordinal_mappings: dict[str, list] | None = None,
    scale_numeric: bool = True,
) -> tuple[Pipeline, FeatureEngineeringReport]:
    """Build an adaptive preprocessing pipeline from the observed training schema."""

    feature_builder = DynamicCreditFeatureBuilder().fit(x_train)
    engineered_train = feature_builder.transform(x_train)
    inferred_ordinals = infer_ordinal_mappings(engineered_train, ordinal_mappings)
    schema = profile_schema(engineered_train, inferred_ordinals)

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", SelectiveOutlierClipper()),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if schema.numeric_columns:
        transformers.append(("numeric", Pipeline(steps=numeric_steps), schema.numeric_columns))
    if schema.boolean_columns:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                schema.boolean_columns,
            )
        )
    if schema.ordinal_columns:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=[
                                    inferred_ordinals[column] for column in schema.ordinal_columns
                                ],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                schema.ordinal_columns,
            )
        )
    if schema.categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                schema.categorical_columns,
            )
        )

    report = FeatureEngineeringReport(
        original_features=list(x_train.columns),
        derived_features=feature_builder.derived_features_,
        dropped_features=schema.dropped_columns + schema.date_columns,
        numeric_columns=schema.numeric_columns,
        categorical_columns=schema.categorical_columns,
        ordinal_columns=schema.ordinal_columns,
        boolean_columns=schema.boolean_columns,
        date_columns=schema.date_columns,
        missing_value_summary=x_train.isna().mean().sort_values(ascending=False).to_dict(),
        encoding_summary={
            "scaled_numeric" if scale_numeric else "unscaled_numeric": schema.numeric_columns,
            "ordinal": schema.ordinal_columns,
            "one_hot": schema.categorical_columns,
            "boolean_ordinal": schema.boolean_columns,
            "date_derived_then_dropped": schema.date_columns,
        },
    )

    return (
        Pipeline(
            steps=[
                ("features", feature_builder),
                ("columns", ColumnTransformer(transformers=transformers, remainder="drop")),
            ]
        ),
        report,
    )
