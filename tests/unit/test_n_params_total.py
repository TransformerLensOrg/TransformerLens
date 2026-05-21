"""Tests for HookedTransformer.n_params_total property.

Related to https://github.com/TransformerLensOrg/TransformerLens/issues/448
where users asked for a total parameter count (including embeddings/biases),
since cfg.n_params only counts the "hidden weight" subset.
"""

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig


def _make_small_attn_only_model() -> HookedTransformer:
    """Build a tiny attn-only HookedTransformer for fast unit tests."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        n_ctx=16,
        d_vocab=50,
        attn_only=True,
    )
    return HookedTransformer(cfg)


def _make_small_mlp_model() -> HookedTransformer:
    """Build a tiny model with MLPs for fast unit tests."""
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=8,
        n_heads=4,
        n_ctx=16,
        d_vocab=50,
        attn_only=False,
        d_mlp=64,
        act_fn="gelu",
    )
    return HookedTransformer(cfg)


def test_n_params_total_attn_only_matches_hand_computed():
    """Verify n_params_total returns the exact expected count for a known small fixture.

    For the 2-layer attn-only model with d_model=32, d_head=8, n_heads=4,
    n_ctx=16, d_vocab=50, the parameter breakdown is:

        embed.W_E:       50 * 32 =  1,600
        pos_embed.W_pos: 16 * 32 =    512
        per block:
            ln1: w + b               = 32 + 32 =       64
            attn.W_Q/K/V/O: 4 * 32 * 8 = 1024 each, 4*1024 = 4,096
            attn.b_Q/K/V:   4 * 8 = 32 each, 3 * 32      =     96
            attn.b_O:       32                            =     32
            block subtotal:                                  4,288
        2 blocks                                            8,576
        ln_final: w + b                                       64
        unembed.W_U: 32 * 50 = 1,600  +  b_U: 50           = 1,650

        TOTAL                                              12,402
    """
    model = _make_small_attn_only_model()
    assert model.n_params_total == 12402


def test_n_params_total_with_mlp_matches_hand_computed():
    """Verify n_params_total for a small model with MLPs.

    Adds to the attn-only count above, per block:
        ln2: w + b                                   =     64
        mlp.W_in:  32 * 64                           =  2,048
        mlp.W_out: 64 * 32                           =  2,048
        mlp.b_in:  64                                =     64
        mlp.b_out: 32                                =     32
        MLP block delta                              =  4,256

    Two MLP blocks add 8,512 to the attn-only 12,402 total = 20,914.
    """
    model = _make_small_mlp_model()
    assert model.n_params_total == 20914


def test_n_params_total_includes_embeddings():
    """n_params_total must be strictly larger than cfg.n_params (which excludes embeddings)."""
    model = _make_small_attn_only_model()
    # cfg.n_params = 8192 (only attention hidden weights)
    # n_params_total = 12402 (full count)
    # Difference proves embeddings/biases/layer norms are included.
    assert model.n_params_total > model.cfg.n_params
    assert model.n_params_total - model.cfg.n_params == 12402 - 8192


def test_n_params_total_returns_int():
    """The property should return a Python int, not a tensor."""
    model = _make_small_attn_only_model()
    assert isinstance(model.n_params_total, int)


def test_n_params_total_real_model_gpt2(gpt2_hooked_processed):
    """End-to-end sanity check on a real loaded model (GPT-2, cached by CI).

    Note: TL's GPT-2 reports more parameters than HuggingFace's because HF ties
    its lm_head with the input embedding (``model.tie_word_embeddings``), while
    TL stores ``W_E`` and ``W_U`` as separate Parameter objects. The expected
    delta is therefore exactly ``d_vocab * d_model`` (one untied projection)
    plus small adjustments from layer-norm folding during ``from_pretrained``.

    This test verifies n_params_total reports the same count produced by
    iterating ``model.parameters()`` on the loaded model — i.e. the property
    correctly reflects what's actually stored.
    """
    tl = gpt2_hooked_processed
    expected = sum(p.numel() for p in tl.parameters())
    assert tl.n_params_total == expected
    # Sanity: GPT-2 is ~124M-163M params depending on tying; ours falls in this band.
    assert 100_000_000 < tl.n_params_total < 200_000_000
