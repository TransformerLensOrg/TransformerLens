"""Tests for padding-side and sequence-axis tokenization semantics."""

from copy import deepcopy
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from transformer_lens import utilities as utils


def _tokenizer(
    bos_token_id: int | None,
    pad_token_id: int | None,
    padding_side: str,
    cls_token_id: int | None = None,
) -> PreTrainedTokenizerBase:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.bos_token_id = bos_token_id
    tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = padding_side
    tokenizer.cls_token_id = cls_token_id
    return cast(PreTrainedTokenizerBase, tokenizer)


def test_attention_mask_uses_explicit_padding_side(gpt2_tokenizer) -> None:
    tokenizer = deepcopy(gpt2_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad = tokenizer.pad_token_id
    tokens = torch.tensor([[pad, pad, 10, 11], [pad, 20, 21, 22]])

    mask = utils.get_attention_mask(tokenizer, tokens, prepend_bos=True, padding_side="left")

    assert torch.equal(mask, torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))


def test_bos_removal_uses_explicit_padding_side(gpt2_tokenizer) -> None:
    tokenizer = deepcopy(gpt2_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.convert_ids_to_tokens(0)
    tokenizer.padding_side = "right"
    pad = tokenizer.pad_token_id
    bos = tokenizer.bos_token_id
    tokens = torch.tensor([[pad, bos, 10, 11], [bos, 20, 21, 22]])

    result = utils.get_tokens_with_bos_removed(tokenizer, tokens, padding_side="left")

    assert torch.equal(result, torch.tensor([[pad, 10, 11], [20, 21, 22]]))


@pytest.mark.parametrize(
    "tokens, expected, bos_token_id, pad_token_id, padding_side",
    [
        pytest.param([0, 1, 11, 12], [0, 11, 12], 1, 0, "left", id="rank-1-left"),
        pytest.param(
            [[0, 1, 11, 12], [1, 21, 22, 23]],
            [[0, 11, 12], [21, 22, 23]],
            1,
            0,
            "left",
            id="rank-2-left-control",
        ),
        pytest.param([1], [], 1, 0, "left", id="bos-only-empty-result"),
        pytest.param([0, 0, 11, 12], [0, 11, 12], 0, 0, "left", id="rank-1-bos-is-pad"),
        pytest.param(
            [[[0, 1, 11, 12], [0, 1, 21, 22]], [[1, 31, 32, 33], [0, 1, 41, 42]]],
            [[[0, 11, 12], [0, 21, 22]], [[31, 32, 33], [0, 41, 42]]],
            1,
            0,
            "left",
            id="rank-3-left",
        ),
        pytest.param(
            [[[0, 0, 0, 11, 12]], [[0, 21, 22, 23, 24]]],
            [[[0, 0, 11, 12]], [[21, 22, 23, 24]]],
            0,
            0,
            "left",
            id="rank-3-bos-is-pad-leading-pads",
        ),
        pytest.param(
            [[[1, 11, 12, 0], [1, 21, 22, 0]], [[1, 31, 32, 33], [1, 41, 42, 0]]],
            [[[11, 12, 0], [21, 22, 0]], [[31, 32, 33], [41, 42, 0]]],
            1,
            0,
            "right",
            id="rank-3-right-control",
        ),
        pytest.param(
            [[[0, 7, 8], [0, 9, 10]]],
            [[[0, 7, 8], [0, 9, 10]]],
            None,
            0,
            "left",
            id="no-bos-unchanged",
        ),
    ],
)
def test_bos_removal_uses_last_axis(
    tokens: list,
    expected: list,
    bos_token_id: int | None,
    pad_token_id: int | None,
    padding_side: str,
) -> None:
    result = utils.get_tokens_with_bos_removed(
        _tokenizer(bos_token_id, pad_token_id, padding_side), torch.tensor(tokens)
    )

    assert torch.equal(result, torch.tensor(expected, dtype=torch.long))


@pytest.mark.parametrize(
    "tokens, bos_token_id, pad_token_id",
    [
        # A BOS-free row has no BOS to drop; column 0 is the real token 21.
        pytest.param([[1, 11, 12], [21, 22, 23]], 1, 0, id="row-without-bos"),
        # The only BOS sits mid-row, so removing it would delete a real in-text token.
        pytest.param([[0, 5, 1, 6]], 1, 0, id="bos-not-leading"),
        # bos == pad with no leading pad, so the second row has no prepended BOS.
        pytest.param([[0, 11, 12], [5, 21, 22]], 0, 0, id="bos-is-pad-no-leading-pad"),
        pytest.param([[0, 0, 0]], 1, 0, id="all-pad-row"),
    ],
)
def test_bos_removal_rejects_left_padded_rows_without_bos(
    tokens: list, bos_token_id: int, pad_token_id: int
) -> None:
    with pytest.raises(ValueError, match="do not start with the BOS token"):
        utils.get_tokens_with_bos_removed(
            _tokenizer(bos_token_id, pad_token_id, "left"), torch.tensor(tokens)
        )


def test_bos_removal_left_padding_drops_leading_cls() -> None:
    """A CLS-prepending tokenizer loses [CLS] under left padding, as it does under right."""
    tokenizer = _tokenizer(bos_token_id=1, pad_token_id=0, padding_side="left", cls_token_id=101)
    tokens = torch.tensor([[0, 101, 11], [101, 21, 22]])

    result = utils.get_tokens_with_bos_removed(tokenizer, tokens)

    assert torch.equal(result, torch.tensor([[0, 11], [21, 22]]))


def _in_memory_tokenizer(bos_token: str = "BOS", **kwargs: Any) -> PreTrainedTokenizerFast:
    """A fast tokenizer built from a tokenizers object, with no hub or disk origin."""
    raw = Tokenizer(
        WordLevel({"UNK": 0, bos_token: 1, "EOS": 2, "hello": 3, "world": 4}, unk_token="UNK")
    )
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, bos_token=bos_token, eos_token="EOS", unk_token="UNK", **kwargs
    )


def test_tokenizer_with_bos_enables_bos_on_in_memory_tokenizer() -> None:
    tokenizer = _in_memory_tokenizer()
    tokenizer.padding_side = "left"

    result = utils.get_tokenizer_with_bos(tokenizer)

    assert result is not tokenizer
    assert result("hello world")["input_ids"] == [1, 3, 4]
    assert result.padding_side == "left"
    assert tokenizer("hello world")["input_ids"] == [3, 4]


def test_tokenizer_with_bos_keeps_in_memory_tokenizer_that_already_adds_bos() -> None:
    tokenizer = _in_memory_tokenizer(add_bos_token=True)

    assert utils.get_tokenizer_with_bos(tokenizer) is tokenizer
    assert tokenizer("hello world")["input_ids"] == [1, 3, 4]


def test_tokenizer_with_bos_returns_in_memory_tokenizer_when_bos_is_inexpressible() -> None:
    # TemplateProcessing rejects special tokens containing ':'.
    tokenizer = _in_memory_tokenizer("<seed:bos>")

    assert utils.get_tokenizer_with_bos(tokenizer) is tokenizer


class _InMemorySlowTokenizer(PreTrainedTokenizer):
    """A whitespace slow tokenizer with no hub or disk origin."""

    def __init__(self) -> None:
        self._vocab = {"UNK": 0, "BOS": 1, "EOS": 2, "hello": 3, "world": 4}
        super().__init__(bos_token="BOS", eos_token="EOS", unk_token="UNK")

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return next(token for token, i in self._vocab.items() if i == index)


def test_tokenizer_with_bos_returns_in_memory_slow_tokenizer_unchanged() -> None:
    # A slow tokenizer has no post-processor, so a copy claiming add_bos_token would still omit BOS.
    tokenizer = _InMemorySlowTokenizer()

    assert utils.get_tokenizer_with_bos(tokenizer) is tokenizer
    assert tokenizer("hello world")["input_ids"] == [3, 4]
