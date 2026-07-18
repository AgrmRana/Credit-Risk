import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk_platform.config.settings import Settings, get_settings
from credit_risk_platform.evaluation.explainability import write_explainability_artifacts
from credit_risk_platform.evaluation.metrics import (
    classification_metrics,
    save_metrics,
    write_validation_plots,
)
from credit_risk_platform.feature_engineering.preprocessing import build_preprocessor
from credit_risk_platform.models.registry import save_model_bundle
from credit_risk_platform.training.data import make_train_test_split
from credit_risk_platform.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _candidate_models(random_state: int) -> dict[str, tuple[Any, dict[str, Any]]]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
            {"classifier__C": uniform(0.05, 8.0), "classifier__class_weight": [None, "balanced"]},
        ),
        "ridge_logistic_regression": (
            LogisticRegression(
                penalty="l2",
                max_iter=2000,
                solver="lbfgs",
                random_state=random_state,
            ),
            {"classifier__C": uniform(0.05, 5.0), "classifier__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "classifier__n_estimators": randint(150, 450),
                "classifier__max_depth": [3, 4, 5, 8, None],
                "classifier__min_samples_leaf": randint(2, 12),
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": randint(80, 260),
                "classifier__max_depth": randint(2, 5),
                "classifier__learning_rate": uniform(0.025, 0.15),
                "classifier__subsample": uniform(0.65, 0.3),
                "classifier__colsample_bytree": uniform(0.65, 0.3),
            },
        ),
    }


def _feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.named_steps["columns"].get_feature_names_out())


def _feature_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    random_state: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    return importance


def train_experiment(settings: Settings | None = None) -> dict[str, Any]:
    configure_logging()
    settings = settings or get_settings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle = make_train_test_split(random_state=settings.random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_state)
    comparison: dict[str, Any] = {}
    best_name = ""
    best_search: RandomizedSearchCV | None = None
    best_auc = -np.inf

    for name, (estimator, params) in _candidate_models(settings.random_state).items():
        LOGGER.info("Training candidate model: %s", name)
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(bundle.x_train)),
                ("classifier", estimator),
            ]
        )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=8,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=settings.random_state,
            refit=True,
        )
        search.fit(bundle.x_train, bundle.y_train)
        cv_scores = cross_validate(
            search.best_estimator_,
            bundle.x_train,
            bundle.y_train,
            scoring=["roc_auc", "average_precision"],
            cv=cv,
            n_jobs=-1,
        )
        y_score = search.predict_proba(bundle.x_test)[:, 1]
        metrics = classification_metrics(bundle.y_test.to_numpy(), y_score)
        metrics["cv_roc_auc_mean"] = float(cv_scores["test_roc_auc"].mean())
        metrics["cv_roc_auc_std"] = float(cv_scores["test_roc_auc"].std())
        metrics["cv_pr_auc_mean"] = float(cv_scores["test_average_precision"].mean())
        metrics["best_params"] = search.best_params_
        comparison[name] = metrics
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_search = search

    if best_search is None:
        raise RuntimeError("No model was trained.")

    champion = best_search.best_estimator_
    champion_scores = champion.predict_proba(bundle.x_test)[:, 1]
    plot_paths = write_validation_plots(
        bundle.y_test.to_numpy(),
        champion_scores,
        settings.reports_dir,
        prefix=best_name,
    )
    feature_importance = _feature_importance(
        champion,
        bundle.x_test,
        bundle.y_test,
        settings.feature_importance_path,
        settings.random_state,
    )
    explainability_artifacts = write_explainability_artifacts(
        champion,
        bundle.x_test,
        settings.reports_dir,
        feature_importance["feature"].head(8).tolist(),
    )

    metrics_report = {
        "dataset": "OpenML credit-g / UCI German Credit",
        "target_definition": "bad credit risk mapped to default=1; good mapped to default=0",
        "champion_model": best_name,
        "model_comparison": comparison,
        "validation_artifacts": plot_paths,
        "explainability_artifacts": explainability_artifacts,
        "top_features": feature_importance.head(10).to_dict(orient="records"),
        "n_train": int(len(bundle.x_train)),
        "n_test": int(len(bundle.x_test)),
        "features": bundle.feature_names,
    }
    save_metrics(metrics_report, settings.metrics_path)
    save_model_bundle(
        {
            "model": champion,
            "model_name": best_name,
            "metrics": metrics_report,
            "feature_names": bundle.feature_names,
            "target_definition": metrics_report["target_definition"],
        },
        settings.model_artifact_path,
    )
    return metrics_report


if __name__ == "__main__":
    report = train_experiment()
    print(f"Champion model: {report['champion_model']}")
    print(f"ROC AUC: {report['model_comparison'][report['champion_model']]['roc_auc']:.4f}")
