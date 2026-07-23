from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Settings:
    """Local application configuration. No env files, no deployment targets."""

    app_name: str = "Credit Risk Decision Platform"
    artifact_dir: Path = field(default_factory=lambda: Path("artifacts"))
    model_artifact_path: Path = field(default_factory=lambda: Path("artifacts/model.joblib"))
    metrics_path: Path = field(default_factory=lambda: Path("artifacts/metrics.json"))
    feature_importance_path: Path = field(
        default_factory=lambda: Path("artifacts/feature_importance.csv")
    )
    reports_dir: Path = field(default_factory=lambda: Path("artifacts/reports"))
    default_dataset: str = "german"
    random_state: int = 42


@lru_cache
def get_settings() -> Settings:
    return Settings()
