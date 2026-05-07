"""Tests for TransformerBridge tokenization methods.

Mirrors the high- and medium-value cases from
``tests/integration/test_tokenization_methods.py`` (HookedTransformer side).
HT-specific cases (set_tokenizer flow, tokens_to_residual_directions) are
omitted. Bridge-specific paths (tokenizer_appends_eos) are added.

Uses the ``distilgpt2_bridge`` session fixture from conftest.py to avoid
per-test model loads.
"""

import pytest
import torch
from torch import equal, tensor


def test_to_tokens_default_prepends_bos(distilgpt2_bridge):
    """Default behavior prepends a BOS token at position 0."""
    bridge = distilgpt2_bridge
    tokens = bridge.to_tokens("Hello, world!")
    assert tokens.dim() == 2 and tokens.shape[0] == 1
    assert tokens[0, 0].item() == bridge.tokenizer.bos_token_id


def test_to_tokens_without_bos_drops_bos(distilgpt2_bridge):
    """``prepend_bos=False`` matches the with-BOS sequence minus the leading BOS."""
    bridge = distilgpt2_bridge
    with_bos = bridge.to_tokens("Hello, world!")
    without_bos = bridge.to_tokens("Hello, world!", prepend_bos=False)
    assert equal(with_bos[0, 1:], without_bos[0])
    assert without_bos[0, 0].item() != bridge.tokenizer.bos_token_id


def test_to_tokens_truncate(distilgpt2_bridge):
    """``truncate=True`` clips to ``cfg.n_ctx``; ``truncate=False`` keeps full length."""
    bridge = distilgpt2_bridge
    n_ctx = bridge.cfg.n_ctx
    long_string = "@ " * (n_ctx + 10)
    truncated = bridge.to_tokens(long_string)
    full = bridge.to_tokens(long_string, truncate=False)
    assert truncated.shape[1] == n_ctx
    assert full.shape[1] > n_ctx


def test_to_string_round_trip_without_bos(distilgpt2_bridge):
    """``to_string(to_tokens(s, prepend_bos=False))`` returns the original string."""
    bridge = distilgpt2_bridge
    s = "Hello, world!"
    tokens = bridge.to_tokens(s, prepend_bos=False)
    decoded = bridge.to_string(tokens[0])
    assert decoded == s


def test_to_string_multiple_returns_list(distilgpt2_bridge):
    """2D token tensor decodes to a list of strings, one per row."""
    bridge = distilgpt2_bridge
    s_a = "Hello, world!"
    s_b = "Goodbye, world!"
    tokens_a = bridge.to_tokens(s_a, prepend_bos=False)
    tokens_b = bridge.to_tokens(s_b, prepend_bos=False)
    # Pad to the same length so we can stack
    max_len = max(tokens_a.shape[1], tokens_b.shape[1])
    pad_id = bridge.tokenizer.pad_token_id or bridge.tokenizer.eos_token_id
    tokens_a = torch.nn.functional.pad(tokens_a, (0, max_len - tokens_a.shape[1]), value=pad_id)
    tokens_b = torch.nn.functional.pad(tokens_b, (0, max_len - tokens_b.shape[1]), value=pad_id)
    stacked = torch.cat([tokens_a, tokens_b], dim=0)
    result = bridge.to_string(stacked)
    assert isinstance(result, list) and len(result) == 2
    assert s_a in result[0] and s_b in result[1]


def test_to_str_tokens_default_starts_with_bos(distilgpt2_bridge):
    """Default ``to_str_tokens`` includes the BOS string token at position 0."""
    bridge = distilgpt2_bridge
    str_tokens = bridge.to_str_tokens("Hello, world!")
    assert str_tokens[0] == bridge.tokenizer.bos_token


def test_to_str_tokens_without_bos_skips_bos(distilgpt2_bridge):
    """``prepend_bos=False`` strips the BOS string token from the front."""
    bridge = distilgpt2_bridge
    with_bos = bridge.to_str_tokens("Hello, world!")
    without_bos = bridge.to_str_tokens("Hello, world!", prepend_bos=False)
    assert with_bos[1:] == without_bos
    assert without_bos[0] != bridge.tokenizer.bos_token


def test_to_single_token(distilgpt2_bridge):
    """A known single-piece word maps to a single token id."""
    bridge = distilgpt2_bridge
    # " the" is a single BPE piece in GPT-2's vocab
    token_id = bridge.to_single_token(" the")
    assert isinstance(token_id, int)
    # Round-trip: decoding the single id returns the same string
    assert bridge.tokenizer.decode([token_id]) == " the"


def test_get_token_position_not_found_raises(distilgpt2_bridge):
    """Asking for a token absent from the input raises AssertionError."""
    bridge = distilgpt2_bridge
    with pytest.raises(AssertionError, match="does not occur"):
        # Use a token id that is not in the input tensor to bypass
        # to_single_token and hit the "does not occur" assertion.
        bridge.get_token_position(99999, tensor([1, 2, 3, 4]))


def test_get_token_position_str_with_bos(distilgpt2_bridge):
    """String input with default ``prepend_bos`` shifts positions by 1 (BOS at index 0)."""
    bridge = distilgpt2_bridge
    pos_with = bridge.get_token_position(" world", "Hello, world!")
    pos_without = bridge.get_token_position(" world", "Hello, world!", prepend_bos=False)
    assert pos_with == pos_without + 1


def test_get_token_position_str_without_bos(distilgpt2_bridge):
    """``prepend_bos=False`` returns the natural in-string position."""
    bridge = distilgpt2_bridge
    pos = bridge.get_token_position(" world", "Hello, world!", prepend_bos=False)
    tokens = bridge.to_tokens("Hello, world!", prepend_bos=False)[0]
    # Verify by indexing rather than re-running the search algorithm.
    assert tokens[pos].item() == bridge.to_single_token(" world")


def test_get_token_position_int_pos_ignores_prepend_bos(distilgpt2_bridge):
    """Tensor input is taken as-is; ``prepend_bos`` has no effect."""
    bridge = distilgpt2_bridge
    target = 100
    input_tokens = tensor([2, 3, target, 5])
    pos_with = bridge.get_token_position(target, input_tokens)
    pos_without = bridge.get_token_position(target, input_tokens, prepend_bos=False)
    assert pos_with == 2 == pos_without


def test_get_token_position_mode_last(distilgpt2_bridge):
    """``mode="last"`` returns the last occurrence index."""
    bridge = distilgpt2_bridge
    target = 7
    input_tokens = tensor([target, 3, 4, target, 5])
    assert bridge.get_token_position(target, input_tokens, mode="first") == 0
    assert bridge.get_token_position(target, input_tokens, mode="last") == 3


def test_get_token_position_2d_tensor_flattens(distilgpt2_bridge):
    """A ``[1, seq]``-shaped tensor is squeezed and indexed correctly."""
    bridge = distilgpt2_bridge
    target = 9
    input_tokens = tensor([[2, target, 4]])
    assert bridge.get_token_position(target, input_tokens) == 1


def test_to_tokens_strips_trailing_eos_when_appends_eos_set(distilgpt2_bridge):
    """``cfg.tokenizer_appends_eos=True`` strips trailing EOS tokens from output.

    Bridge-specific path covering OLMo / Apertus tokenizers that auto-append
    EOS — the model expects continuations, not terminated sequences.
    """
    bridge = distilgpt2_bridge
    original = getattr(bridge.cfg, "tokenizer_appends_eos", False)
    eos_str = bridge.tokenizer.eos_token
    eos_id = bridge.tokenizer.eos_token_id
    try:
        # Baseline: strip disabled, trailing EOS is preserved.
        bridge.cfg.tokenizer_appends_eos = False
        with_eos = bridge.to_tokens(f"hello {eos_str}", prepend_bos=False)
        assert with_eos[0, -1].item() == eos_id

        # Strip enabled: trailing EOS removed; everything else identical.
        bridge.cfg.tokenizer_appends_eos = True
        stripped = bridge.to_tokens(f"hello {eos_str}", prepend_bos=False)
        assert stripped.shape[1] == with_eos.shape[1] - 1
        assert equal(stripped[0], with_eos[0, :-1])
        assert stripped[0, -1].item() != eos_id
    finally:
        bridge.cfg.tokenizer_appends_eos = original
