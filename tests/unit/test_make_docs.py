"""Make Docs Tests."""
import pytest

from transformer_lens import HookedTransformerConfig
from transformer_lens.make_docs import get_config, get_property


def test_get_config():
    """Test get config with attn-only-1l model."""
    config: HookedTransformerConfig = get_config("attn-only-1l")
    assert config.attn_only is True


def test_get_property():
    """Test get property with attn-only-1l model."""
    # Test act_fn property
    act_fn = get_property("act_fn", "attn-only-1l")
    assert act_fn == "attn_only"

    # Test n_params property
    n_params = get_property("n_params", "attn-only-1l")
    assert n_params == "1.0M"

    # Test n_layers property
    n_layers = get_property("n_layers", "attn-only-1l")
    assert n_layers == 1

    # Test d_model property
    d_model = get_property("d_model", "attn-only-1l")
    assert d_model == 512

    # Test n_heads property
    n_heads = get_property("n_heads", "attn-only-1l")
    assert n_heads == 8

    # Test n_ctx property
    n_ctx = get_property("n_ctx", "attn-only-1l")
    assert n_ctx == 1024

    # Test d_vocab property
    d_vocab = get_property("d_vocab", "attn-only-1l")
    assert d_vocab == 48262

    # Test d_head property
    d_head = get_property("d_head", "attn-only-1l")
    assert d_head == 64

    # Test d_mlp property
    d_mlp = get_property("d_mlp", "attn-only-1l")
    assert d_mlp == 2048

    # Test an unknown property
    with pytest.raises(KeyError):
        get_property("unknown_property", "attn-only-1l")
