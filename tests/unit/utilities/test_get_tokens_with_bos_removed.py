"""Tests for get_tokens_with_bos_removed when the tokenizer has no BOS token.

Callers gate this helper on ``cfg.tokenizer_prepends_bos``. That flag is set by
``detect_tokenizer_bos_eos``, which requires a ``bos_token_id`` — so a tokenizer
with none should never reach here. It does when the flag is stale: a bridge built
via ``build_bridge_from_module(tokenizer=None)`` keeps the config default of True,
and the tokenizer setter only re-runs detection on *re*-assignment.

Trusting a stale flag is not harmless. Under right padding the helper drops the
first token unconditionally, which silently removes ``[CLS]`` from a BERT
tokenizer's output; under left padding it compares tokens against ``None`` and
raises an ``AttributeError`` naming neither the tokenizer nor the flag.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from transformer_lens.utilities.tokenize_utils import get_tokens_with_bos_removed


@pytest.fixture(scope="module")
def no_bos_tokenizer():
    """BERT uses [CLS] rather than a BOS token, so bos_token_id is None."""
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    assert tokenizer.bos_token_id is None
    return tokenizer


@pytest.fixture(scope="module")
def bos_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    assert tokenizer.bos_token_id is not None
    return tokenizer


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_no_bos_token_returns_tokens_unchanged(no_bos_tokenizer, padding_side) -> None:
    """There is no BOS to remove, so the tokens must come back untouched."""
    no_bos_tokenizer.padding_side = padding_side
    tokens = torch.tensor([[101, 19082, 1362, 102]])

    result = get_tokens_with_bos_removed(no_bos_tokenizer, tokens)

    torch.testing.assert_close(result, tokens)


def test_no_bos_token_does_not_drop_cls_under_right_padding(no_bos_tokenizer) -> None:
    """The damaging case: [CLS] is not a BOS token, and dropping it changes what
    the model is asked to encode."""
    no_bos_tokenizer.padding_side = "right"
    tokens = no_bos_tokenizer("hello world", return_tensors="pt")["input_ids"]

    result = get_tokens_with_bos_removed(no_bos_tokenizer, tokens)

    assert result.shape == tokens.shape
    assert result[0, 0].item() == no_bos_tokenizer.cls_token_id


def test_no_bos_token_does_not_raise_under_left_padding(no_bos_tokenizer) -> None:
    """Previously `(tokens == None).int()` — a Python bool, not a tensor."""
    no_bos_tokenizer.padding_side = "left"
    tokens = torch.tensor([[101, 19082, 1362, 102]])

    result = get_tokens_with_bos_removed(no_bos_tokenizer, tokens)

    assert result.shape == tokens.shape


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_a_real_bos_is_still_removed(bos_tokenizer, padding_side) -> None:
    """The guard must not disturb the case the helper exists for."""
    bos_tokenizer.padding_side = padding_side
    bos = bos_tokenizer.bos_token_id
    tokens = torch.tensor([[bos, 15496, 995]])

    result = get_tokens_with_bos_removed(bos_tokenizer, tokens)

    assert result.shape[-1] == tokens.shape[-1] - 1
    assert bos not in result[0].tolist()
