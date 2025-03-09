from transformer_lens.supported_models import MODEL_ALIASES, OFFICIAL_MODEL_NAMES


def test_official_model_names_is_alphabetical():
    assert OFFICIAL_MODEL_NAMES == sorted(
        OFFICIAL_MODEL_NAMES, key=str.casefold
    ), "OFFICIAL_MODEL_NAMES are not alphabetical"


def test_model_aliases_is_alphabetical():
    # Extract the keys as they appear in the dictionary
    actual_keys = list(MODEL_ALIASES.keys())

    # Create a sorted version, ignoring case
    expected_keys = sorted(actual_keys, key=str.casefold)

    # Compare the actual insertion order to the expected alphabetical order
    assert actual_keys == expected_keys, "MODEL_ALIASES keys are not in alphabetical order. "
