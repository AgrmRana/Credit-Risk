import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringReport:
    original_features: list[str]
    derived_features: list[str]
    dropped_features: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    ordinal_columns: list[str]
    boolean_columns: list[str]
    date_columns: list[str]
    missing_value_summary: dict[str, float]
    encoding_summary: dict[str, list[str]]


@dataclass(frozen=True)
class SchemaProfile:
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    ordinal_columns: list[str] = field(default_factory=list)
    boolean_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)


ORDERED_VALUE_SETS = [
    ["none", "low", "medium", "high", "very high"],
    ["unknown", "poor", "fair", "good", "excellent"],
    ["a", "b", "c", "d", "e", "f", "g"],
]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def matches_any(name: str, patterns: list[str]) -> bool:
    normalized = normalize_name(name)
    return any(re.search(pattern, normalized) for pattern in patterns)


def infer_semantic_column(
    columns: list[str],
    include: list[str],
    exclude: list[str] | None = None,
) -> str | None:
    exclude = exclude or []
    for column in columns:
        normalized = normalize_name(column)
        if any(re.search(pattern, normalized) for pattern in include) and not any(
            re.search(pattern, normalized) for pattern in exclude
        ):
            return column
    return None


def infer_semantic_columns(
    columns: list[str],
    include: list[str],
    exclude: list[str] | None = None,
) -> list[str]:
    exclude = exclude or []
    selected = []
    for column in columns:
        normalized = normalize_name(column)
        if any(re.search(pattern, normalized) for pattern in include) and not any(
            re.search(pattern, normalized) for pattern in exclude
        ):
            selected.append(column)
    return selected


def _looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False
    non_missing = series.dropna().astype(str).head(100)
    if non_missing.empty:
        return False
    parsed = pd.to_datetime(non_missing, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.8)


def _looks_ordinal_by_values(values: list[Any]) -> list[Any] | None:
    normalized_values = [str(value).strip().lower() for value in values]
    for ordered_values in ORDERED_VALUE_SETS:
        if set(normalized_values).issubset(set(ordered_values)):
            return [value for value in ordered_values if value in normalized_values]
    return None


def infer_ordinal_mappings(
    frame: pd.DataFrame,
    configured_mappings: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    mappings = dict(configured_mappings or {})
    for column in frame.columns:
        if column in mappings:
            continue
        if not (
            pd.api.types.is_object_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        ):
            continue
        unique_values = frame[column].dropna().unique().tolist()
        if 2 <= len(unique_values) <= 8 and matches_any(
            column,
            [r"grade", r"rating", r"status", r"risk", r"quality", r"level", r"band"],
        ):
            inferred = _looks_ordinal_by_values(unique_values)
            if inferred:
                mappings[column] = inferred
    return mappings


def profile_schema(
    frame: pd.DataFrame,
    ordinal_mappings: dict[str, list[Any]] | None = None,
) -> SchemaProfile:
    ordinal_mappings = ordinal_mappings or {}
    dropped_columns = [
        column
        for column in frame.columns
        if frame[column].isna().mean() >= 0.98 or frame[column].nunique() <= 1
    ]
    working = frame.drop(columns=dropped_columns)

    date_columns = [column for column in working.columns if _looks_like_date(working[column])]
    boolean_columns = [
        column
        for column in working.columns
        if column not in date_columns
        and (
            pd.api.types.is_bool_dtype(working[column])
            or set(working[column].dropna().unique()).issubset(
                {0, 1, "0", "1", "Y", "N", "yes", "no"}
            )
        )
    ]
    ordinal_columns = [column for column in ordinal_mappings if column in working.columns]
    numeric_columns = [
        column
        for column in working.select_dtypes(include="number").columns
        if column not in boolean_columns
    ]
    categorical_columns = [
        column
        for column in working.columns
        if column not in set(numeric_columns + boolean_columns + ordinal_columns + date_columns)
    ]
    return SchemaProfile(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        ordinal_columns=ordinal_columns,
        boolean_columns=boolean_columns,
        date_columns=date_columns,
        dropped_columns=dropped_columns,
    )
