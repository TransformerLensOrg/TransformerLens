import torch

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def test_atten_result_normal_attn_correct():
    """Verifies that the attn_result flag does not change the output for models with normal attention."""
    d_model = 128
    d_head = 8
    n_heads = 16
    n_ctx = 128
    n_layers = 1
    d_vocab = 10

    cfg = HookedTransformerConfig(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_ctx=n_ctx,
        n_layers=n_layers,
        attn_only=True,
        d_vocab=d_vocab,
    )

    model = HookedTransformer(cfg)
    assert model.cfg.use_split_qkv_input is False

    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = model(x)

    model.set_use_attn_result(True)
    assert model.cfg.use_attn_result is True

    split_output = model(x)

    assert torch.allclose(normal_output, split_output, atol=1e-6)


def test_atten_result_grouped_query_attn_correct():
    """Verifies that the atten_result flag does not change the output for models with grouped query attention."""

    d_model = 128
    d_head = 8
    n_heads = 16
    n_ctx = 128
    n_key_value_heads = 2
    n_layers = 1
    d_vocab = 10

    cfg = HookedTransformerConfig(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_ctx=n_ctx,
        n_key_value_heads=n_key_value_heads,
        n_layers=n_layers,
        attn_only=True,
        d_vocab=d_vocab,
    )

    model = HookedTransformer(cfg)
    assert model.cfg.use_split_qkv_input is False

    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = model(x)

    model.set_use_attn_result(True)
    assert model.cfg.use_attn_result is True

    split_output = model(x)

    assert torch.allclose(normal_output, split_output, atol=1e-6)
