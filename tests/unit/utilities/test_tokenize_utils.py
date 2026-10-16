"""Tests for padding-side and sequence-axis tokenization semantics."""

from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from transformer_lens import utilities as utils


def _tokenizer(
    bos_token_id: int | None,
    pad_token_id: int | None,
    padding_side: str,
) -> PreTrainedTokenizerBase:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.bos_token_id = bos_token_id
    tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = padding_side
    return cast(PreTrainedTokenizerBase, tokenizer)


def test_attention_mask_uses_explicit_padding_side(gpt2_tokenizer) -> None:
    tokenizer = deepcopy(gpt2_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad = tokenizer.pad_token_id
    tokens = torch.tensor([[pad, pad, 10, 11], [pad, 20, 21, 22]])

    mask = utils.get_attention_mask(tokenizer, tokens, prepend_bos=True, padding_side="left")

    assert torch.equal(mask, torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_attention_mask_without_pad_id_attends_every_token(padding_side, dtype) -> None:
    tokenizer = _tokenizer(bos_token_id=7, pad_token_id=None, padding_side=padding_side)
    tokens = torch.tensor([[7, 8, 9], [3, 4, 5]], dtype=dtype)

    mask = utils.get_attention_mask(tokenizer, tokens, prepend_bos=True)

    torch.testing.assert_close(mask, torch.ones_like(tokens))


def test_attention_mask_preserves_bos_when_bos_is_pad() -> None:
    tokenizer = _tokenizer(bos_token_id=0, pad_token_id=0, padding_side="left")
    tokens = torch.tensor([[0, 0, 10, 11], [0, 20, 21, 22]])

    mask = utils.get_attention_mask(tokenizer, tokens, prepend_bos=True)

    assert torch.equal(mask, torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))


def test_attention_mask_masks_trailing_pads_on_right() -> None:
    tokenizer = _tokenizer(bos_token_id=1, pad_token_id=0, padding_side="right")
    tokens = torch.tensor([[1, 10, 0, 0], [1, 20, 21, 22]])

    mask = utils.get_attention_mask(tokenizer, tokens, prepend_bos=True)

    assert torch.equal(mask, torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]))


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
