from unittest import mock

from transformer_lens.loading_from_pretrained import fill_missing_keys
from transformer_lens.model_bridge import TransformerBridge


def get_transformer_bridge():
    """Get a TransformerBridge for testing."""
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


# Successes


@mock.patch("logging.warning")
def test_fill_missing_keys(mock_warning: mock.MagicMock):
    model = get_transformer_bridge()
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
    model = get_transformer_bridge()
    default_state_dict = model.state_dict()

    incomplete_state_dict = {k: v for k, v in default_state_dict.items() if "hf_model" not in k}

    filled_state_dict = fill_missing_keys(model, incomplete_state_dict)

    expected_keys = set(default_state_dict.keys()) - {
        k for k in default_state_dict.keys() if "hf_model" in k
    }
    assert set(filled_state_dict.keys()) == expected_keys


def test_fill_missing_keys_no_missing_keys():
    model = get_transformer_bridge()
    default_state_dict = model.state_dict()

    filled_state_dict = fill_missing_keys(model, default_state_dict)

    assert filled_state_dict == default_state_dict
