from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_model_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle["saved_at"] = datetime.now(UTC).isoformat()
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run training first.")
    return joblib.load(path)
