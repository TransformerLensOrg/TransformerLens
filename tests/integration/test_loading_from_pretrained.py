"""
Tests that verify than an arbitrary component (e.g. Embed) can be initialized using dict and object versions of HookedTransformerConfig and HookedEncoderConfig.
"""

from transformer_lens import loading_from_pretrained as loading


def test_get_basic_config():
    cfg = loading.get_basic_config("gpt2-small")
    assert cfg.d_model
    assert cfg.layer_norm_eps
    assert cfg.d_vocab
    assert cfg.init_range
    assert cfg.n_ctx
    assert cfg.d_head
    assert cfg.d_mlp
    assert cfg.n_heads
    assert cfg.n_layers
