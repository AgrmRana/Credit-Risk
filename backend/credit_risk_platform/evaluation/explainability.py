import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def write_explainability_artifacts(
    model: Any,
    x_reference: pd.DataFrame,
    output_dir: Path,
    top_features: list[str],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    numeric_columns = x_reference.select_dtypes(include="number").columns
    pdp_features = [feature for feature in top_features if feature in numeric_columns][:3]
    if pdp_features:
        try:
            fig, ax = plt.subplots(figsize=(9, 4))
            PartialDependenceDisplay.from_estimator(
                model,
                x_reference.astype({column: float for column in numeric_columns}),
                features=pdp_features,
                kind="average",
                ax=ax,
            )
            path = output_dir / "partial_dependence.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            artifacts["partial_dependence"] = str(path)
        except Exception as exc:  # pragma: no cover - artifact best effort
            LOGGER.warning("Partial dependence generation failed: %s", exc)

    try:
        import shap

        sample = x_reference.sample(min(50, len(x_reference)), random_state=42)
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(sample)
        transformed_frame = pd.DataFrame(
            transformed,
            columns=preprocessor.named_steps["columns"].get_feature_names_out(),
        )

        def predict_pd(frame: np.ndarray) -> np.ndarray:
            return classifier.predict_proba(np.asarray(frame))[:, 1]

        explainer = shap.Explainer(predict_pd, transformed_frame)
        values = explainer(transformed_frame)
        shap.summary_plot(values, transformed_frame, plot_type="bar", show=False, max_display=12)
        path = output_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts["shap_summary"] = str(path)
    except Exception as exc:  # pragma: no cover - artifact best effort
        LOGGER.warning("SHAP generation failed: %s", exc)

    return artifacts
