"""Tests for get_input_with_manually_prepended_bos when the tokenizer has no BOS token.

``to_tokens`` reaches this helper whenever the caller wants a BOS that the tokenizer
will not add on its own — ``prepend_bos and not cfg.tokenizer_prepends_bos``. For a
tokenizer with no BOS token that condition is *correctly* true rather than stale:
``detect_tokenizer_bos_eos`` requires a ``bos_token_id``, so it reports False for
BERT and T5, and ``prepend_bos`` defaults to True. The helper then evaluated
``None + input`` and raised ``TypeError: unsupported operand type(s) for +:
'NoneType' and 'str'``, naming neither the tokenizer nor the flag.

This is the prepend-side counterpart of #1628, which covers the removal side.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from transformer_lens.utilities.tokenize_utils import (
    get_input_with_manually_prepended_bos,
)


@pytest.fixture(
    scope="module",
    params=["google-bert/bert-base-cased", "google-t5/t5-small"],
)
def no_bos_tokenizer(request):
    """BERT opens with [CLS] and T5 with nothing, so bos_token is None for both."""
    tokenizer = AutoTokenizer.from_pretrained(request.param)
    assert tokenizer.bos_token is None
    return tokenizer


@pytest.fixture(scope="module")
def bos_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    assert tokenizer.bos_token is not None
    return tokenizer


def test_no_bos_token_returns_string_unchanged(no_bos_tokenizer) -> None:
    """There is no BOS to prepend, so the string must come back untouched."""
    assert get_input_with_manually_prepended_bos(no_bos_tokenizer.bos_token, "hello world") == (
        "hello world"
    )


def test_no_bos_token_returns_list_unchanged(no_bos_tokenizer) -> None:
    """Same for the batched form — and no partially-prepended list."""
    inputs = ["hello world", "second string"]

    result = get_input_with_manually_prepended_bos(no_bos_tokenizer.bos_token, inputs)

    assert result == ["hello world", "second string"]


def test_a_real_bos_is_still_prepended_to_a_string(bos_tokenizer) -> None:
    """The guard must not disturb the case the helper exists for."""
    result = get_input_with_manually_prepended_bos(bos_tokenizer.bos_token, "hello world")

    assert result == bos_tokenizer.bos_token + "hello world"


def test_a_real_bos_is_still_prepended_to_a_list(bos_tokenizer) -> None:
    bos = bos_tokenizer.bos_token

    result = get_input_with_manually_prepended_bos(bos, ["hello world", "second string"])

    assert result == [bos + "hello world", bos + "second string"]
