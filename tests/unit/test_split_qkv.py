import torch

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig


def test_split_qkv_normal_attn_correct():
    """Enabling split_qkv_input activates the per-head q/k/v_input hooks (sized at n_heads) without changing the output for normal attention."""
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

    model.set_use_split_qkv_input(True)

    split_hooks = [
        "blocks.0.hook_q_input",
        "blocks.0.hook_k_input",
        "blocks.0.hook_v_input",
    ]
    split_output, cache = model.run_with_cache(x, names_filter=split_hooks)

    # The flag's effect: the per-head fork hooks now fire, each carrying a head
    # dimension equal to n_heads (normal attention has no separate KV grouping).
    for name in split_hooks:
        assert name in cache, f"{name} did not fire after enabling use_split_qkv_input"
        assert cache[name].shape[2] == n_heads

    assert torch.allclose(normal_output, split_output, atol=1e-6)


def test_split_qkv_grouped_query_attn_correct():
    """Enabling split_qkv_input forks the q_input hook at n_heads but the k/v_input hooks at n_key_value_heads for grouped query attention, without changing the output."""

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

    model.set_use_split_qkv_input(True)

    split_hooks = [
        "blocks.0.hook_q_input",
        "blocks.0.hook_k_input",
        "blocks.0.hook_v_input",
    ]
    split_output, cache = model.run_with_cache(x, names_filter=split_hooks)

    # GQA-specific effect: the query fork keeps n_heads, but the key/value forks
    # are sized at the smaller n_key_value_heads, not n_heads.
    assert cache["blocks.0.hook_q_input"].shape[2] == n_heads
    assert cache["blocks.0.hook_k_input"].shape[2] == n_key_value_heads
    assert cache["blocks.0.hook_v_input"].shape[2] == n_key_value_heads

    assert torch.allclose(normal_output, split_output, atol=1e-6)
