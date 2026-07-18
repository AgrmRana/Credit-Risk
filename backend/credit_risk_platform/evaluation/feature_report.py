from html import escape
from pathlib import Path

import pandas as pd

from credit_risk_platform.feature_engineering.schema import FeatureEngineeringReport


def _list_section(values: list[str]) -> str:
    if not values:
        return "- None\n"
    return "\n".join(f"- `{value}`" for value in values) + "\n"


def write_feature_report(
    report: FeatureEngineeringReport,
    feature_importance: pd.DataFrame,
    explainability_artifacts: dict[str, str],
    output_dir: Path,
    dataset_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{dataset_name}_feature_report.md"
    html_path = output_dir / f"{dataset_name}_feature_report.html"

    top_missing = {
        key: value for key, value in list(report.missing_value_summary.items())[:20] if value > 0
    }
    top_importance = feature_importance.head(20)
    missing_markdown = (
        pd.Series(top_missing, name="missing_rate").to_markdown()
        if top_missing
        else "No missing values detected in the top missingness summary."
    )
    shap_artifact = explainability_artifacts.get(
        "shap_summary",
        "SHAP artifact was not generated.",
    )
    pdp_artifact = explainability_artifacts.get(
        "partial_dependence",
        "Partial dependence artifact was not generated.",
    )
    original_items = "".join(
        f"<li><code>{escape(value)}</code></li>" for value in report.original_features
    )
    derived_items = "".join(
        f"<li><code>{escape(value)}</code></li>" for value in report.derived_features
    )
    dropped_items = "".join(
        f"<li><code>{escape(value)}</code></li>" for value in report.dropped_features
    )
    missing_html = (
        pd.Series(top_missing, name="missing_rate").to_frame().to_html()
        if top_missing
        else "<p>No missing values detected in the top missingness summary.</p>"
    )
    artifact_items = "".join(
        f"<li>{escape(key)}: <code>{escape(value)}</code></li>"
        for key, value in explainability_artifacts.items()
    )

    markdown = f"""# Feature Engineering Report: {dataset_name}

## Original Features

{_list_section(report.original_features)}
## Derived Features

{_list_section(report.derived_features)}
## Dropped Features

Dropped features include unusable constant/high-missingness columns and raw date columns
after date-derived variables are created.

{_list_section(report.dropped_features)}
## Missing Value Summary

{missing_markdown}

## Categorical Encoding Summary

- Numeric columns: {len(report.numeric_columns)}
- Boolean columns: {len(report.boolean_columns)}
- Ordinal encoded columns: {len(report.ordinal_columns)}
- One-hot encoded columns: {len(report.categorical_columns)}
- Date columns transformed then dropped: {len(report.date_columns)}

## Feature Importance

{top_importance.to_markdown(index=False)}

## SHAP Summary

{shap_artifact}

## Partial Dependence

{pdp_artifact}
"""

    markdown_path.write_text(markdown, encoding="utf-8")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Feature Engineering Report: {escape(dataset_name)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      color: #18202b;
      line-height: 1.45;
    }}
    h1, h2 {{ color: #0f172a; }}
    code {{ background: #eef2f6; padding: 2px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border-bottom: 1px solid #d9dee7; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Feature Engineering Report: {escape(dataset_name)}</h1>
  <h2>Original Features</h2>
  <ul>{original_items}</ul>
  <h2>Derived Features</h2>
  <ul>{derived_items or '<li>None</li>'}</ul>
  <h2>Dropped Features</h2>
  <ul>{dropped_items or '<li>None</li>'}</ul>
  <h2>Missing Value Summary</h2>
  {missing_html}
  <h2>Categorical Encoding Summary</h2>
  <ul>
    <li>Numeric columns: {len(report.numeric_columns)}</li>
    <li>Boolean columns: {len(report.boolean_columns)}</li>
    <li>Ordinal encoded columns: {len(report.ordinal_columns)}</li>
    <li>One-hot encoded columns: {len(report.categorical_columns)}</li>
    <li>Date columns transformed then dropped: {len(report.date_columns)}</li>
  </ul>
  <h2>Feature Importance</h2>
  {top_importance.to_html(index=False)}
  <h2>Explainability Artifacts</h2>
  <ul>{artifact_items}</ul>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return {"markdown": str(markdown_path), "html": str(html_path)}
