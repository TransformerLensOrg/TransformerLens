from unittest import mock

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import fill_missing_keys


def get_default_config():
    return HookedTransformerConfig(
        d_model=128, d_head=8, n_heads=16, n_ctx=128, n_layers=1, d_vocab=50257, attn_only=True
    )


# Successes


@mock.patch("logging.warning")
def test_fill_missing_keys(mock_warning: mock.MagicMock):
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


def test_n_ctx_override_reduces_context():
    """
    n_ctx override should work when reducing below the model default.
    Uses a minimal HookedTransformerConfig directly — no model loading needed.
    Fixes #1006.
    """
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    cfg = get_pretrained_model_config("gpt2", n_ctx=256)
    assert cfg.n_ctx == 256, f"Expected n_ctx=256, got {cfg.n_ctx}"


@mock.patch("logging.warning")
def test_n_ctx_override_larger_than_default_warns(mock_warning: mock.MagicMock):
    """
    A warning should be issued when n_ctx exceeds the model's default.
    GPT-2 default n_ctx is 1024 — requesting 2048 should trigger the warning.
    Fixes #1006.
    """
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    cfg = get_pretrained_model_config("gpt2", n_ctx=2048)
    assert cfg.n_ctx == 2048, f"Expected n_ctx=2048, got {cfg.n_ctx}"
    mock_warning.assert_any_call(
        "You are setting n_ctx=2048 which is larger than this model's "
        "default context length of 1024. The model was not "
        "trained on sequences this long and may produce unreliable results. "
        "Ensure you have sufficient memory for this context length."
    )


# --- Architecture config tests ---


class TestArchitectureConfigs:
    """Verify that convert_hf_model_config produces correct configs for new architectures."""

    def test_apertus_config(self):
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        cfg = get_pretrained_model_config("apertus-8b")
        assert cfg.original_architecture == "ApertusForCausalLM"
        assert cfg.normalization_type == "RMS"
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is False
        assert cfg.final_rms is True
        assert cfg.act_fn == "xielu"
        assert cfg.use_qk_norm is True
        assert cfg.n_key_value_heads is not None
        assert cfg.d_model > 0
        assert cfg.n_heads > 0

    def test_gpt_oss_config(self):
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        cfg = get_pretrained_model_config("gpt-oss-20b")
        assert cfg.original_architecture == "GptOssForCausalLM"
        assert cfg.normalization_type == "RMS"
        assert cfg.positional_embedding_type == "rotary"
        assert cfg.gated_mlp is True
        assert cfg.final_rms is True
        assert cfg.num_experts is not None
        assert cfg.num_experts > 0
        assert cfg.experts_per_token is not None
        assert cfg.n_key_value_heads is not None

    def test_apertus_instruct_config(self):
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        cfg = get_pretrained_model_config("apertus-8b-instruct")
        assert cfg.original_architecture == "ApertusForCausalLM"
        assert cfg.act_fn == "xielu"
