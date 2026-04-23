"""Tests for HookedTransformer.generate() when no tokenizer is set.

Regression test for https://github.com/TransformerLensOrg/TransformerLens/issues/483
"""

import torch

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def _make_tokenizer_free_model() -> HookedTransformer:
    """Create a small HookedTransformer with no tokenizer (algorithmic task setup)."""
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=16,
        d_head=4,
        n_heads=4,
        n_ctx=32,
        d_vocab=20,
        attn_only=True,
    )
    return HookedTransformer(cfg)


def test_generate_without_tokenizer_stop_at_eos_false():
    """generate() with raw token tensors and stop_at_eos=False should not require a tokenizer."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    output = model.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_explicit_eos_token_id():
    """generate() with an explicit eos_token_id should not require a tokenizer."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    output = model.generate(
        tokens,
        max_new_tokens=5,
        stop_at_eos=True,
        eos_token_id=0,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape[0] == 1
    assert output.shape[1] >= 5