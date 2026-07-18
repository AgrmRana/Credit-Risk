import os
from dataclasses import dataclass

import certifi
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

TARGET = "default"


@dataclass(frozen=True)
class DatasetBundle:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def load_german_credit() -> tuple[pd.DataFrame, pd.Series]:
    """Load the public German Credit dataset from OpenML.

    OpenML dataset ``credit-g`` is a canonical version of the UCI German Credit
    dataset. Target values are mapped as bad=1 and good=0 to represent default.
    """

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    frame = fetch_openml(name="credit-g", version=1, as_frame=True).frame
    target = frame.pop("class").map({"good": 0, "bad": 1}).astype(int)
    return frame, target.rename(TARGET)


def make_train_test_split(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    x, y = load_german_credit()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetBundle(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(x.columns),
    )
