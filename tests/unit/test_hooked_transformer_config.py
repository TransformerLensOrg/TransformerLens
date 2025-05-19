"""
Tests that config passed around TransformerLens can be unwrapped into an actual configuration object
"""

from transformer_lens.TransformerLensConfig import TransformerLensConfig


def test_transformer_lens_config_object():
    config = TransformerLensConfig(
        n_layers=2, d_vocab=100, d_model=6, n_ctx=5, d_head=2
    )
    result = TransformerLensConfig.from_dict(config.to_dict())
    # Assert that the config was properly converted
    assert isinstance(result, TransformerLensConfig)
    assert result.n_layers == config.n_layers
    assert result.d_vocab == config.d_vocab
    assert result.d_model == config.d_model
    assert result.n_ctx == config.n_ctx
    assert result.d_head == config.d_head


def test_transformer_lens_config_dict():
    config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
    }
    result = TransformerLensConfig.from_dict(config_dict)
    # Assert that the new returned value has been transformed into a config object
    assert isinstance(result, TransformerLensConfig)
    assert result.n_layers == config_dict["n_layers"]
    assert result.d_vocab == config_dict["d_vocab"]
    assert result.d_model == config_dict["d_model"]
    assert result.n_ctx == config_dict["n_ctx"]
    assert result.d_head == config_dict["d_head"]


def test_is_layer_norm_activation_passes():
    config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
        "act_fn": "solu_ln",
    }
    config = TransformerLensConfig.from_dict(config_dict)
    assert config.is_layer_norm_activation()


def test_is_layer_norm_activation_fails():
    config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
        "act_fn": "relu",
    }
    config = TransformerLensConfig.from_dict(config_dict)
    assert not config.is_layer_norm_activation()
