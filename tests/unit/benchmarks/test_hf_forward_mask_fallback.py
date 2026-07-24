"""`_hf_forward_with_mask_fallback` must feed a mask to models that demand one.

LLaDA2's diffusion block attention dereferences ``attention_mask.size()`` and
rejects both ``None`` and a 2D mask, requiring 4D ``(b,1,s,s)``. The Phase-1 HF
capture previously called ``hf_model(tokens)`` bare, caught the resulting
``NoneType`` error, and silently degraded to a shape-only forward check — which
is how a genuinely-divergent model could still read as verified.
"""

from types import SimpleNamespace

import torch

from transformer_lens.benchmarks.main_benchmark import _hf_forward_with_mask_fallback

TOKENS = torch.tensor([[1, 2, 3, 4]])


def test_plain_model_needs_no_mask() -> None:
    """A normal decoder that accepts bare input_ids is called exactly once."""
    calls = []

    def model(tokens, attention_mask=None):
        calls.append(attention_mask)
        return SimpleNamespace(logits=torch.zeros(1, 4, 8))

    out = _hf_forward_with_mask_fallback(model, TOKENS)
    assert out.logits.shape == (1, 4, 8)
    assert calls == [None]  # no fallback triggered


def test_model_requiring_4d_mask_gets_it() -> None:
    """LLaDA-like: rejects None and 2D, accepts 4D (b,1,s,s)."""
    seen = []

    def model(tokens, attention_mask=None):
        seen.append(None if attention_mask is None else tuple(attention_mask.shape))
        if attention_mask is None:
            raise AttributeError("'NoneType' object has no attribute 'size'")
        if attention_mask.dim() != 4:
            raise ValueError("only supports block attention mask (b,1,s,s)")
        return SimpleNamespace(logits=torch.ones(1, 4, 8))

    out = _hf_forward_with_mask_fallback(model, TOKENS)
    assert out.logits.shape == (1, 4, 8)
    # tried bare, then 2D, then succeeded with 4D
    assert seen == [None, (1, 4), (1, 1, 4, 4)]


def test_unrelated_error_still_propagates() -> None:
    """The fallback must not swallow a genuine forward failure."""
    def model(tokens, attention_mask=None):
        raise ValueError("real bug unrelated to masking")

    try:
        _hf_forward_with_mask_fallback(model, TOKENS)
    except ValueError as e:
        assert "real bug" in str(e)
    else:
        raise AssertionError("expected the underlying error to propagate")
