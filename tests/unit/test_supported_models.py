from transformer_lens.supported_models import (
    MODEL_ALIASES,
    OFFICIAL_MODEL_NAMES,
    get_official_model_name,
)

OLMO3_BASE_MODELS = {
    "allenai/Olmo-3-1025-7B": "olmo-3-1025-7b",
    "allenai/Olmo-3-1125-32B": "olmo-3-1125-32b",
}


def test_official_model_names_is_alphabetical():
    assert OFFICIAL_MODEL_NAMES == sorted(
        OFFICIAL_MODEL_NAMES, key=str.casefold
    ), "OFFICIAL_MODEL_NAMES are not alphabetical"


def test_model_aliases_is_alphabetical():
    actual_keys = list(MODEL_ALIASES.keys())

    expected_keys = sorted(actual_keys, key=str.casefold)

    assert actual_keys == expected_keys, "MODEL_ALIASES keys are not in alphabetical order. "


def test_get_official_model_name_is_case_insensitive():
    """The deleted loading_from_pretrained resolver lowercased both sides; the
    rehomed one must too, or ~900 previously-accepted case variants regress."""
    assert get_official_model_name("GPT2") == "gpt2"
    assert get_official_model_name("gpt2-small") == get_official_model_name("GPT2-Small")


def test_get_official_model_name_raises_on_unknown():
    import pytest

    with pytest.raises(ValueError, match="not an official model name"):
        get_official_model_name("definitely-not-a-real-model-xyz")


def test_olmo3_base_models_have_supported_aliases():
    for model_name, alias in OLMO3_BASE_MODELS.items():
        assert model_name in OFFICIAL_MODEL_NAMES
        assert alias in MODEL_ALIASES[model_name]
        assert get_official_model_name(model_name) == model_name
        assert get_official_model_name(alias) == model_name
