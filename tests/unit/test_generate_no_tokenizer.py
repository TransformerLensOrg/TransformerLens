"""Tests for HookedTransformer.generate() when no tokenizer is set.

Regression test for https://github.com/TransformerLensOrg/TransformerLens/issues/483
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig


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


def test_generate_without_tokenizer_stop_at_eos_false_kv_cache():
    """generate() with no tokenizer, stop_at_eos=False, use_past_kv_cache=True."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    output = model.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=False,
        use_past_kv_cache=True,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_stop_at_eos_false_no_kv_cache():
    """generate() with no tokenizer, stop_at_eos=False, use_past_kv_cache=False."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    output = model.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=False,
        use_past_kv_cache=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_without_tokenizer_explicit_eos_kv_cache():
    """generate() with no tokenizer, explicit eos_token_id, use_past_kv_cache=True."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    output = model.generate(
        tokens,
        max_new_tokens=5,
        stop_at_eos=True,
        eos_token_id=0,
        use_past_kv_cache=True,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape[0] == 1 and output.shape[1] >= 5


def test_generate_without_tokenizer_explicit_eos_no_kv_cache():
    """generate() with no tokenizer, explicit eos_token_id, use_past_kv_cache=False."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    output = model.generate(
        tokens,
        max_new_tokens=5,
        stop_at_eos=True,
        eos_token_id=0,
        use_past_kv_cache=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape[0] == 1 and output.shape[1] >= 5


def test_generate_without_tokenizer_stop_at_eos_requires_eos_id():
    """generate() must still error when stop_at_eos=True, no eos_token_id, no tokenizer."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    tokens = torch.zeros((1, 5), dtype=torch.long)
    try:
        model.generate(
            tokens, max_new_tokens=3, stop_at_eos=True, return_type="tokens", verbose=False
        )
        raise AssertionError("Should have raised AssertionError")
    except AssertionError as e:
        assert "eos_token_id" in str(e), f"Unexpected error message: {e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_generate_cpu_tokens_cuda_model():
    """generate() with CPU token input and the model on CUDA must not mix devices.

    Regression test for https://github.com/TransformerLensOrg/TransformerLens/issues/1508:
    input_tokens used to stay on the input's original device after the input was
    moved to the model's device, breaking the sampling-path and output torch.cat.
    """
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=16,
        d_head=4,
        n_heads=4,
        n_ctx=32,
        d_vocab=20,
        attn_only=True,
        device="cuda",
    )
    model = HookedTransformer(cfg)

    tokens = torch.zeros((1, 5), dtype=torch.long)  # deliberately left on CPU

    # Sampling path: concatenates input_tokens with sampled tokens each step.
    output = model.generate(
        tokens,
        max_new_tokens=3,
        stop_at_eos=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"

    # Greedy path: concatenates input_tokens with sampled tokens for the output.
    output = model.generate(
        tokens,
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        return_type="tokens",
        verbose=False,
    )
    assert output.shape == (1, 8), f"Expected shape (1, 8), got {output.shape}"


def test_generate_string_input_without_tokenizer_errors():
    """generate() must still error when string input is used without a tokenizer."""
    model = _make_tokenizer_free_model()
    assert model.tokenizer is None

    try:
        model.generate("hello", max_new_tokens=3, verbose=False)
        raise AssertionError("Should have raised AssertionError")
    except AssertionError:
        pass  # Expected
