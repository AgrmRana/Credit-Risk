# Model Governance

## Scope

This document describes governance considerations for the Credit Risk Decision Platform. The current committed model artifact is trained on the public OpenML `credit-g` German Credit dataset and is intended as an internal analytics demonstration, not a production credit approval model.

## Model Assumptions

- The target definition maps observed bad credit risk to default and good credit risk to non-default.
- The training sample is assumed to be representative of the scored population only for demonstration purposes.
- Input variables are assumed to be collected consistently between training and serving.
- Missing values are handled through fitted preprocessing rules and are not treated as independent evidence unless encoded in the raw dataset.
- Business thresholds are policy controls and are intentionally separated from model fitting.

## Limitations

- German Credit is a small public dataset and does not reflect the scale, heterogeneity, or regulatory complexity of a modern bank portfolio.
- Some model families may perform differently on larger datasets such as Home Credit Default Risk.
- Public datasets may lack protected-class attributes, bureau-depth variables, and post-origination performance windows required for real-world validation.
- The tool does not persist prediction history between runs, so it does not yet implement a model-release approval workflow or immutable audit trails.
- The platform does not currently implement reject inference, adverse action reason codes, or fairness monitoring.

## Potential Bias

Credit-risk data can encode historical lending policy, access-to-credit disparities, and socioeconomic proxies. Bias may enter through:

- Historical labels that reflect prior underwriting decisions.
- Variables correlated with age, employment, income, geography, or household structure.
- Missingness patterns that differ across applicant segments.
- Threshold policies that optimize portfolio performance while creating uneven review or rejection rates.

Before production use, the model should be evaluated for disparate impact, segment-level calibration, error-rate differences, and stability across relevant legally approved monitoring segments.

## Validation Methodology

The current validation framework includes:

- Stratified train/test split.
- Cross-validation during model comparison.
- ROC AUC and PR AUC.
- KS statistic and Gini coefficient.
- Precision, recall, F1, and confusion matrix.
- Calibration curve.
- Lift and gain charts.
- Threshold optimization.
- Permutation feature importance.
- SHAP summary plots.
- Partial dependence plots where appropriate.

Validation artifacts are generated from actual trained models and saved under `artifacts/` and `docs/images/`.

## Monitoring Strategy

Recommended production monitoring:

- Input schema validation and missing-value rate checks.
- Population stability index for key variables and score distributions.
- Data drift checks for continuous and categorical variables.
- Calibration monitoring using observed default outcomes when available.
- Segment-level approval, manual-review, and rejection rates.
- Feature importance and explanation drift.
- Prediction volume and error rates if the tool is wrapped in a production serving layer.

## Retraining Recommendations

Retraining should be considered when:

- Score distribution drift exceeds approved stability thresholds.
- Calibration materially deteriorates.
- Portfolio policy, macroeconomic conditions, or product mix changes.
- New performance labels become available.
- Feature availability or collection logic changes.
- A challenger model demonstrates statistically meaningful and governance-approved improvement.

Every retraining event should produce a model card, validation report, approval record, and rollback plan before promotion.

## Production Approval Checklist

- Data lineage reviewed.
- Feature definitions approved.
- Validation metrics independently reproduced.
- Explainability reviewed by model risk and business stakeholders.
- Thresholds approved by credit policy.
- Monitoring dashboards configured.
- Rollback artifact available.
- Security and access controls reviewed.
