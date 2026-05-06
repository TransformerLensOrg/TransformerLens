"""Tests for HookedTransformer.n_params_total property.

Related to https://github.com/TransformerLensOrg/TransformerLens/issues/448
where users asked for a total parameter count (including embeddings/biases),
since cfg.n_params only counts the "hidden weight" subset.
"""

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig


def _make_small_model(attn_only: bool = True, gated_mlp: bool = False) -> HookedTransformer:
    """Build a tiny HookedTransformer for fast unit tests."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        n_ctx=16,
        d_vocab=50,
        attn_only=attn_only,
        d_mlp=64 if not attn_only else None,
        act_fn="gelu" if not attn_only else None,
        gated_mlp=gated_mlp,
    )
    return HookedTransformer(cfg)


def test_n_params_total_matches_sum_of_parameters():
    """n_params_total must equal sum(p.numel() for p in model.parameters())."""
    model = _make_small_model()
    expected = sum(p.numel() for p in model.parameters())
    assert model.n_params_total == expected


def test_n_params_total_includes_embeddings():
    """n_params_total must be strictly larger than cfg.n_params (which excludes embeddings)."""
    model = _make_small_model()
    # cfg.n_params counts only attention projections (no embeddings, no biases, no layer norms).
    # n_params_total counts everything, so it must be larger for any non-trivial model.
    assert model.n_params_total > model.cfg.n_params


def test_n_params_total_with_mlp():
    """n_params_total works for models with MLPs too."""
    model = _make_small_model(attn_only=False)
    expected = sum(p.numel() for p in model.parameters())
    assert model.n_params_total == expected
    assert model.n_params_total > model.cfg.n_params


def test_n_params_total_returns_int():
    """The property should return a Python int, not a tensor."""
    model = _make_small_model()
    assert isinstance(model.n_params_total, int)
