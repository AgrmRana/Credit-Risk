# Feature Engineering Report: german

## Original Features

- `checking_status`
- `duration`
- `credit_history`
- `purpose`
- `credit_amount`
- `savings_status`
- `employment`
- `installment_commitment`
- `personal_status`
- `other_parties`
- `residence_since`
- `property_magnitude`
- `age`
- `other_payment_plans`
- `housing`
- `existing_credits`
- `job`
- `num_dependents`
- `own_telephone`
- `foreign_worker`

## Derived Features

- `credit_amount_per_duration`
- `age_squared`
- `age_band`
- `credit_exposure_score`

## Dropped Features

Dropped features include unusable constant/high-missingness columns and raw date columns after date-derived variables are created.

- None

## Missing Value Summary

No missing values detected in the top missingness summary.

## Categorical Encoding Summary

- Numeric columns: 11
- Boolean columns: 1
- Ordinal encoded columns: 3
- One-hot encoded columns: 9
- Date columns transformed then dropped: 0

## Feature Importance

| feature                |   importance_mean |   importance_std |
|:-----------------------|------------------:|-----------------:|
| checking_status        |       0.1255      |      0.0338156   |
| duration               |       0.0454881   |      0.0114786   |
| credit_amount          |       0.0397619   |      0.0179842   |
| credit_history         |       0.0297024   |      0.0115366   |
| purpose                |       0.0145119   |      0.00564227  |
| age                    |       0.0115595   |      0.00703539  |
| other_parties          |       0.0082619   |      0.00401499  |
| own_telephone          |       0.00667857  |      0.00455455  |
| other_payment_plans    |       0.0054881   |      0.00580958  |
| savings_status         |       0.00521429  |      0.00753183  |
| residence_since        |       0.0049881   |      0.00170321  |
| installment_commitment |       0.002       |      0.00389553  |
| employment             |       0.00170238  |      0.0061882   |
| foreign_worker         |       0.000857143 |      0.000756304 |
| job                    |       0.000654762 |      0.00337611  |
| personal_status        |       0.000285714 |      0.00788365  |
| housing                |      -0.00022619  |      0.00519093  |
| property_magnitude     |      -0.000261905 |      0.00694667  |
| num_dependents         |      -0.000630952 |      0.00143159  |
| existing_credits       |      -0.00222619  |      0.0051113   |

## SHAP Summary

artifacts/reports/shap_summary.png

## Partial Dependence

artifacts/reports/partial_dependence.png
