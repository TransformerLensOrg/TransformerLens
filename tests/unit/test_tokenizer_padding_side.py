"""Tests for tokenizer padding_side preservation.

Regression test for https://github.com/TransformerLensOrg/TransformerLens/issues/801

When a tokenizer is reloaded via AutoTokenizer.from_pretrained() inside
get_tokenizer_with_bos(), HuggingFace silently resets padding_side to its
default (usually "right"). This previously caused user-set padding_side="left"
to be silently discarded.
"""

import pytest
from transformers import AutoTokenizer

from transformer_lens.utilities.tokenize_utils import get_tokenizer_with_bos


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """A fresh GPT-2 tokenizer (cached by HF). add_bos_token defaults to False."""
    return AutoTokenizer.from_pretrained("gpt2")


def test_padding_side_left_is_preserved(gpt2_tokenizer):
    """User-set padding_side='left' must be preserved through get_tokenizer_with_bos()."""
    gpt2_tokenizer.padding_side = "left"
    result = get_tokenizer_with_bos(gpt2_tokenizer)
    assert (
        result.padding_side == "left"
    ), f"padding_side='left' was reset to '{result.padding_side}' after reload"


def test_padding_side_right_is_preserved(gpt2_tokenizer):
    """User-set padding_side='right' must also round-trip correctly."""
    gpt2_tokenizer.padding_side = "right"
    result = get_tokenizer_with_bos(gpt2_tokenizer)
    assert result.padding_side == "right"


def test_padding_side_preserved_when_already_has_bos():
    """When add_bos_token is already True, function returns original tokenizer unchanged."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_bos_token=True)
    tokenizer.padding_side = "left"
    result = get_tokenizer_with_bos(tokenizer)
    # Same object (no reload happened), so padding_side is naturally preserved
    assert result.padding_side == "left"
