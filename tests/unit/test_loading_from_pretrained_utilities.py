from unittest import mock

import pytest

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import fill_missing_keys


def get_default_config():
    return HookedTransformerConfig(
        d_model=128, d_head=8, n_heads=16, n_ctx=128, n_layers=1, d_vocab=50257, attn_only=True
    )


# Successes


@mock.patch("logging.warning")
def test_fill_missing_keys(mock_warning):
    cfg = get_default_config()
    model = HookedTransformer(cfg)
    default_state_dict = model.state_dict()

    incomplete_state_dict = {k: v for k, v in default_state_dict.items() if "W_" not in k}

    filled_state_dict = fill_missing_keys(model, incomplete_state_dict)

    assert set(filled_state_dict.keys()) == set(default_state_dict.keys())

    # Check that warnings were issued for missing weight matrices
    for key in default_state_dict:
        if "W_" in key and key not in incomplete_state_dict:
            mock_warning.assert_any_call(
                f"Missing key for a weight matrix in pretrained, filled in with an empty tensor: {key}"
            )


def test_fill_missing_keys_with_hf_model_keys():
    cfg = get_default_config()
    model = HookedTransformer(cfg)
    default_state_dict = model.state_dict()

    incomplete_state_dict = {k: v for k, v in default_state_dict.items() if "hf_model" not in k}

    filled_state_dict = fill_missing_keys(model, incomplete_state_dict)

    expected_keys = set(default_state_dict.keys()) - {
        k for k in default_state_dict.keys() if "hf_model" in k
    }
    assert set(filled_state_dict.keys()) == expected_keys


def test_fill_missing_keys_no_missing_keys():
    cfg = get_default_config()
    model = HookedTransformer(cfg)
    default_state_dict = model.state_dict()

    filled_state_dict = fill_missing_keys(model, default_state_dict)

    assert filled_state_dict == default_state_dict


# Failures


def test_fill_missing_keys_raises_error_on_invalid_model():
    invalid_model = None
    default_state_dict = {}

    with pytest.raises(AttributeError):
        fill_missing_keys(invalid_model, default_state_dict)
