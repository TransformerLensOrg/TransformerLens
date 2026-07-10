from transformer_lens.loading_from_pretrained import get_official_model_name
from transformer_lens.supported_models import MODEL_ALIASES, OFFICIAL_MODEL_NAMES

OLMO3_BASE_MODELS = {
    "allenai/Olmo-3-1025-7B": "olmo-3-1025-7b",
    "allenai/Olmo-3-1125-32B": "olmo-3-1125-32b",
}


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


def test_olmo3_base_models_have_supported_aliases():
    for model_name, alias in OLMO3_BASE_MODELS.items():
        assert model_name in OFFICIAL_MODEL_NAMES
        assert alias in MODEL_ALIASES[model_name]
        assert get_official_model_name(model_name) == model_name
        assert get_official_model_name(alias) == model_name
