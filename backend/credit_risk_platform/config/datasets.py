from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Minimal dataset metadata required by the training pipeline."""

    name: str
    display_name: str
    target_column: str
    source_type: str
    openml_name: str | None = None
    openml_version: int | None = None
    csv_path: Path | None = None
    target_mapping: dict[Any, int] | None = None
    ordinal_mappings: dict[str, list[Any]] = field(default_factory=dict)
    ignored_columns: list[str] = field(default_factory=list)


GERMAN_ORDINALS = {
    "checking_status": ["no checking", "<0", "0<=X<200", ">=200"],
    "savings_status": ["no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
}


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "german": DatasetConfig(
        name="german",
        display_name="OpenML credit-g / UCI German Credit",
        target_column="class",
        source_type="openml",
        openml_name="credit-g",
        openml_version=1,
        target_mapping={"good": 0, "bad": 1},
        ordinal_mappings=GERMAN_ORDINALS,
    ),
    "give_me_some_credit": DatasetConfig(
        name="give_me_some_credit",
        display_name="Give Me Some Credit",
        target_column="SeriousDlqin2yrs",
        source_type="csv",
        csv_path=Path("data/raw/give_me_some_credit/cs-training.csv"),
        ignored_columns=["Unnamed: 0"],
    ),
    "home_credit": DatasetConfig(
        name="home_credit",
        display_name="Home Credit Default Risk",
        target_column="TARGET",
        source_type="csv",
        csv_path=Path("data/raw/home_credit/application_train.csv"),
        ignored_columns=["SK_ID_CURR"],
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    try:
        return DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}"
        ) from exc
