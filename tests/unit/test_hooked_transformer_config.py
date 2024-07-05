"""
Tests that config passed around TransformerLens can be unwrapped into an actual configuration object
"""

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def test_hooked_transformer_config_object():
    hooked_transformer_config = HookedTransformerConfig(
        n_layers=2, d_vocab=100, d_model=6, n_ctx=5, d_head=2, attn_only=True
    )
    result = HookedTransformerConfig.unwrap(hooked_transformer_config)
    # Assert that the same object was returned
    assert result is hooked_transformer_config


def test_hooked_transformer_config_dict():
    hooked_transformer_config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
    }
    result = HookedTransformerConfig.unwrap(hooked_transformer_config_dict)
    # Assert that the new returned value has been transformed into a config object
    assert isinstance(result, HookedTransformerConfig)


def test_is_layer_norm_activation_passes():
    hooked_transformer_config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
        "act_fn": "solu_ln",
    }
    config = HookedTransformerConfig.unwrap(hooked_transformer_config_dict)
    assert config.is_layer_norm_activation()


def test_is_layer_norm_activation_fails():
    hooked_transformer_config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
        "act_fn": "relu",
    }
    config = HookedTransformerConfig.unwrap(hooked_transformer_config_dict)
    assert not config.is_layer_norm_activation()
