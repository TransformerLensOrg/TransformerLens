import torch

from transformer_lens.components import AbstractAttention, Attention
from transformer_lens.config import HookedTransformerConfig


def test_attention_clip_qkv_clamps_between_projection_and_scores():
    """cfg.clip_qkv must clamp Q/K/V after projection (OLMo v1 / OLMoE semantics):
    output with clip_qkv equals a clip-free module fed pre-clamped Q/K/V."""
    cfg_kwargs = dict(n_layers=1, d_model=32, n_ctx=8, d_head=8, n_heads=4, act_fn="relu")
    # fork_rng: deterministic setup without mutating the global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        attn_clip = Attention(HookedTransformerConfig(**cfg_kwargs, clip_qkv=0.5))
        attn_ref = Attention(HookedTransformerConfig(**cfg_kwargs))
        # Standalone components carry torch.empty weights (init_weights runs at
        # model level), so fill them explicitly before sharing the state dict.
        with torch.no_grad():
            for param in attn_clip.parameters():
                param.normal_(0, 0.02)
        attn_ref.load_state_dict(attn_clip.state_dict())

        batch, seq = 1, 4
        q0 = torch.randn(batch, seq, 4, 8) * 2
        k0 = torch.randn(batch, seq, 4, 8) * 2
        v0 = torch.randn(batch, seq, 4, 8) * 2
        x = torch.randn(batch, seq, 32)
    # clamp is active on all three of Q/K/V
    assert (q0.abs() > 0.5).any() and (k0.abs() > 0.5).any() and (v0.abs() > 0.5).any()

    attn_clip.calculate_qkv_matrices = lambda *args, **kwargs: (q0, k0, v0)
    attn_ref.calculate_qkv_matrices = lambda *args, **kwargs: (
        q0.clamp(min=-0.5, max=0.5),
        k0.clamp(min=-0.5, max=0.5),
        v0.clamp(min=-0.5, max=0.5),
    )

    with torch.no_grad():
        out_clip = attn_clip(x, x, x)
        out_ref = attn_ref(x, x, x)

    torch.testing.assert_close(out_clip, out_ref)


def test_create_alibi_slope():
    n_ctx = 100

    # Expected result computed non-vectorized way
    expected = torch.zeros((n_ctx, n_ctx))
    for row in range(n_ctx):
        for col in range(n_ctx):
            expected[row, col] = float(min(col - row, 0))

    # Check against the method's vectorized version
    result = AbstractAttention.create_alibi_slope(n_ctx)
    assert torch.allclose(expected, result)


def test_create_alibi_bias():
    n_heads = 2
    n_ctx = 4

    result = AbstractAttention.create_alibi_bias(n_heads, n_ctx, torch.device("cpu"))

    for matrix in result:
        n_row, n_col = matrix.size()
        slope = -matrix[1, 0]
        # Check if upper triangle is all zeros
        assert torch.equal(torch.triu(matrix), torch.zeros_like(matrix))

        ref_lower_triangle = torch.zeros_like(matrix)
        for i in range(1, n_row):
            for j in range(i):
                ref_lower_triangle[i, j] = -slope * (i - j)

        # Check if the lower triangle is decreasing by a constant slope (towards the bottom left corner).
        assert torch.equal(
            torch.tril(matrix, diagonal=-1), torch.tril(ref_lower_triangle, diagonal=-1)
        )
