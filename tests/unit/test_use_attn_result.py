import torch

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig


def test_atten_result_normal_attn_correct():
    """Enabling use_attn_result exposes per-head hook_result that sums back to attn output (normal attention)."""
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

    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = model(x)

    model.set_use_attn_result(True)

    split_output, cache = model.run_with_cache(x)

    # Switching to the per-head compute branch must not change the output.
    assert torch.allclose(normal_output, split_output, atol=1e-6)

    result = cache["blocks.0.attn.hook_result"]
    assert result.shape == (x.shape[0], x.shape[1], n_heads, d_model)
    summed = result.sum(dim=2) + model.blocks[0].attn.b_O
    assert torch.allclose(summed, cache["blocks.0.hook_attn_out"], atol=1e-5)


def test_atten_result_grouped_query_attn_correct():
    """Enabling use_attn_result exposes per-head hook_result that sums back to attn output (grouped-query attention)."""

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

    x = torch.arange(1, 9).unsqueeze(0)
    normal_output = model(x)

    model.set_use_attn_result(True)

    split_output, cache = model.run_with_cache(x)

    # Switching to the per-head compute branch must not change the output.
    assert torch.allclose(normal_output, split_output, atol=1e-6)

    # fires with shape [batch, pos, n_heads, d_model] (one row per query head even
    # under GQA), and head-sum (plus b_O) reconstructs the attention block output.
    result = cache["blocks.0.attn.hook_result"]
    assert result.shape == (x.shape[0], x.shape[1], n_heads, d_model)
    summed = result.sum(dim=2) + model.blocks[0].attn.b_O
    assert torch.allclose(summed, cache["blocks.0.hook_attn_out"], atol=1e-5)
