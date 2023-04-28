"""
Tests that verify than an arbitrary component (e.g. Embed) can be initialized using dict and object versions of HookedTransformerConfig and HookedEncoderConfig.
"""

import pytest
from transformer_lens.HookedEncoderConfig import HookedEncoderConfig
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components import Embed


def test_hooked_transformer_config_object():
    hooked_transformer_config = HookedTransformerConfig(
        n_layers=2, d_vocab=100, d_model=6, n_ctx=5, d_head=2, attn_only=True
    )
    Embed(hooked_transformer_config)


def test_hooked_transformer_config_dict():
    hooked_transformer_config_dict = {
        "n_layers": 2,
        "d_vocab": 100,
        "d_model": 6,
        "n_ctx": 5,
        "d_head": 2,
        "attn_only": True,
    }
    Embed(hooked_transformer_config_dict)


def test_hooked_encoder_config_object():
    hooked_transformer_config = HookedEncoderConfig(d_model=6, d_vocab=100)
    Embed(hooked_transformer_config)


def test_hooked_encoder_config_dict():
    hooked_encoder_config_dict = {
        "d_model": 6,
        "d_vocab": 100,
        "model_type": "hooked_encoder",
    }
    Embed(hooked_encoder_config_dict)


def test_hooked_encoder_config_dict_requires_model_type():
    # fails because model_type defaults to "hooked_transformer", which has more required arguments
    with pytest.raises(TypeError) as e:
        Embed({"d_model": 6, "d_vocab": 100})

    # Depending how the test is run, a different error witll be raised
    # See https://github.com/neelnanda-io/TransformerLens/issues/190
    dataclass_pattern = r"HookedTransformerConfig.__init__\(\) missing \d+ required positional arguments"
    typeguard_pattern = r"missing a required argument: .*"

    assert e.match(f"{dataclass_pattern}|{typeguard_pattern}")