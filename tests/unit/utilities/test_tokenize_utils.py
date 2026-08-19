"""Tests for per-call padding-side overrides in tokenization utilities."""

from copy import deepcopy

import torch

from transformer_lens import utils


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
