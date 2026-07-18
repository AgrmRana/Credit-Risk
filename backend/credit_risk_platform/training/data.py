import os
from dataclasses import dataclass

import certifi
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from credit_risk_platform.config.datasets import DatasetConfig, get_dataset_config

TARGET = "default"


@dataclass(frozen=True)
class DatasetBundle:
    dataset_config: DatasetConfig
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def _read_source(config: DatasetConfig) -> pd.DataFrame:
    if config.source_type == "openml":
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        if config.openml_name is None or config.openml_version is None:
            raise ValueError(f"OpenML dataset '{config.name}' is missing OpenML metadata.")
        return fetch_openml(
            name=config.openml_name,
            version=config.openml_version,
            as_frame=True,
        ).frame

    if config.source_type == "csv":
        if config.csv_path is None:
            raise ValueError(f"CSV dataset '{config.name}' is missing a csv_path.")
        if not config.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset '{config.name}' was not found at {config.csv_path}. "
                "Place the public dataset CSV at the configured path or update the dataset config."
            )
        return pd.read_csv(config.csv_path)

    raise ValueError(f"Unsupported dataset source_type '{config.source_type}'.")


def load_credit_dataset(
    dataset_name: str = "german",
) -> tuple[pd.DataFrame, pd.Series, DatasetConfig]:
    """Load a configured credit-risk dataset without encoding dataset-specific logic downstream."""

    config = get_dataset_config(dataset_name)
    frame = _read_source(config)
    if config.target_column not in frame.columns:
        raise ValueError(
            f"Target column '{config.target_column}' is not present in dataset '{config.name}'."
        )

    ignored_columns = [column for column in config.ignored_columns if column in frame.columns]
    frame = frame.drop(columns=ignored_columns)
    target = frame.pop(config.target_column)
    if config.target_mapping:
        target = target.map(config.target_mapping)
    target = target.astype(int).rename(TARGET)
    return frame, target, config


def make_train_test_split(
    dataset_name: str = "german",
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    x, y, config = load_credit_dataset(dataset_name)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetBundle(
        dataset_config=config,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(x.columns),
    )
