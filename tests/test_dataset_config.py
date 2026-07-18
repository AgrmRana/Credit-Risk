from credit_risk_platform.config.datasets import get_dataset_config


def test_supported_dataset_configs_are_minimal() -> None:
    german = get_dataset_config("german")
    give_me_some_credit = get_dataset_config("give_me_some_credit")
    home_credit = get_dataset_config("home_credit")

    assert german.target_column == "class"
    assert give_me_some_credit.target_column == "SeriousDlqin2yrs"
    assert home_credit.target_column == "TARGET"
    assert home_credit.ignored_columns == ["SK_ID_CURR"]
