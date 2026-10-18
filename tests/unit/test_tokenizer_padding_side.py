"""Tests for tokenizer padding_side preservation.

Regression test for https://github.com/TransformerLensOrg/TransformerLens/issues/801

When a tokenizer is reloaded via AutoTokenizer.from_pretrained() inside
get_tokenizer_with_bos(), HuggingFace silently resets padding_side to its
default (usually "right"). This previously caused user-set padding_side="left"
to be silently discarded.
"""

import re

import pytest
from transformers import AutoTokenizer

from transformer_lens import utilities
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
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


def test_padding_side_preserved_when_already_has_bos():
    """When add_bos_token is already True, function returns original tokenizer unchanged."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_bos_token=True)
    tokenizer.padding_side = "left"
    result = get_tokenizer_with_bos(tokenizer)
    # Same object (no reload happened), so padding_side is naturally preserved
    assert result.padding_side == "left"


def test_test_prompt_multiple_answers_ignore_tokenizer_padding_side(capsys):
    """test_prompt must rank each answer's first token, not a pad token, under left padding."""
    cfg = TransformerBridgeConfig(
        d_model=32, d_head=16, n_heads=2, n_layers=1, n_ctx=32, d_vocab=50257, seed=0
    )
    model = TransformerBridge.boot_native(cfg, tokenizer=AutoTokenizer.from_pretrained("gpt2"))
    prompt, answers = "Why did the elephant cross the", ["road", "super highway"]
    outputs = {}
    for side in ("right", "left"):
        model.tokenizer.padding_side = side
        utilities.test_prompt(prompt, answers, model, print_details=False)
        outputs[side] = capsys.readouterr().out

    utilities.test_prompt(prompt, "road", model, print_details=False)
    single_answer_rank = re.search(r"\(' road', (\d+)\)", capsys.readouterr().out)

    assert single_answer_rank is not None
    assert outputs["left"] == outputs["right"]
    assert f"(' road', {single_answer_rank.group(1)})" in outputs["left"]
