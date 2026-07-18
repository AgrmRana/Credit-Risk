from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Credit Risk Decision Platform"
    environment: str = "development"
    database_url: str = Field(
        default="sqlite:///./artifacts/credit_risk.db",
        description="Use postgresql+psycopg://user:password@host:5432/dbname in production.",
    )
    artifact_dir: Path = Path("artifacts")
    model_artifact_path: Path = Path("artifacts/model.joblib")
    metrics_path: Path = Path("artifacts/metrics.json")
    feature_importance_path: Path = Path("artifacts/feature_importance.csv")
    reports_dir: Path = Path("artifacts/reports")
    random_state: int = 42


@lru_cache
def get_settings() -> Settings:
    return Settings()
