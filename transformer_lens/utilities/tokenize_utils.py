"""tokenize_utils.

This module contains utility functions related to tokenization
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import einops
import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from transformer_lens.utilities.hf_utils import keep_single_column
from transformer_lens.utilities.tensors import get_cumsum_along_dim


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """Tokenize each document, join with token-level EOS between docs, and reshape into ``(batch, sequence_length)`` rows.

    Useful for training language models on a large text corpus without per-doc
    truncation or padding. Absolute-position-embedding models also benefit by
    avoiding early-token bias (e.g. news articles starting with "CNN").

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (PreTrainedTokenizerBase): The tokenizer. Must have ``bos_token_id`` and ``eos_token_id``.
        streaming (bool, optional): If True, avoids parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): Whether to prepend ``bos_token_id`` to each output row. Defaults to True.

    Returns:
        Dataset: Tokenized dataset of tensors with a single column ``"tokens"``.
    """
    dataset = keep_single_column(dataset, column_name)
    has_pad_token = tokenizer.pad_token is not None
    if not has_pad_token:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = max_length - 1 if add_bos_token else max_length

    # Long docs legitimately exceed model_max_length; we slice into rows after.
    _deprecation_warnings_saved = None
    if hasattr(tokenizer, "deprecation_warnings"):
        _deprecation_warnings_saved = tokenizer.deprecation_warnings.copy()
        tokenizer.deprecation_warnings[
            "sequence-length-is-longer-than-the-specified-maximum"
        ] = False

    def tokenize_function(examples: Any) -> dict[str, np.ndarray]:
        text = examples[column_name]
        assert tokenizer.eos_token is not None, "Tokenizer must have an EOS token."
        if not text:
            return {"tokens": np.array([], dtype=np.int64)}

        # Per-doc tokenization with explicit token-level EOS — string chunking
        # could cut tokens mid-doc (#1133); add_special_tokens=False prevents
        # SentencePiece tokenizers from scattering auto-BOS/EOS per call.
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        pieces: list[np.ndarray] = []
        for i, row in enumerate(encoded):
            pieces.append(np.asarray(row, dtype=np.int64))
            if i < len(encoded) - 1:
                pieces.append(np.array([eos_id], dtype=np.int64))
        if not pieces:
            return {"tokens": np.array([], dtype=np.int64)}
        tokens = np.concatenate(pieces)
        num_tokens = len(tokens)

        if num_tokens < seq_len:
            num_batches = 1
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                # Pad with EOS when no native pad token to avoid OOV IDs.
                padding_id = tokenizer.eos_token_id if not has_pad_token else tokenizer.pad_token_id
                tokens = np.concatenate(
                    [tokens, np.full(seq_len - len(tokens), padding_id)], axis=0
                )
        else:
            num_batches = num_tokens // seq_len
            tokens = tokens[: seq_len * num_batches]

        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=(num_proc if not streaming else None),
            remove_columns=[column_name],
        )
    finally:
        if _deprecation_warnings_saved is not None:
            tokenizer.deprecation_warnings.clear()
            tokenizer.deprecation_warnings.update(_deprecation_warnings_saved)
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


def get_tokenizer_with_bos(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """
    Returns the tokenizer initialized with add_bos_token=True.
    Such a tokenizer should be set as the default tokenizer because the tokenization of some
    tokenizers like LlamaTokenizer are different when bos token is automatically/manually
    prepended.

    Note: For tokenizers without a BOS token (e.g., T5), this returns the original tokenizer
    unchanged since add_bos_token=True would fail in transformers v5+ when bos_token is None.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to initialize with add_bos_token=True.

    Returns:
        PreTrainedTokenizerBase: The tokenizer initialized with add_bos_token=True,
            or the original tokenizer if it has no BOS token.
    """
    # If the tokenizer has no BOS token, we can't set add_bos_token=True
    # This is the case for T5 and other encoder-decoder models
    if tokenizer.bos_token is None:
        return tokenizer

    init_kwargs = deepcopy(tokenizer.init_kwargs)
    pretrained_model_name_or_path = init_kwargs.pop("name_or_path")
    add_bos_token = init_kwargs.pop("add_bos_token", None)
    if add_bos_token is None:
        add_bos_token = getattr(tokenizer, "add_bos_token", False)

    if add_bos_token:
        tokenizer_with_bos = tokenizer
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        tokenizer_with_bos = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            add_bos_token=True,
            token=huggingface_token if len(huggingface_token) > 0 else None,
            **init_kwargs,
        )

    return tokenizer_with_bos


def get_input_with_manually_prepended_bos(
    bos_token: str, input: str | list[str]
) -> str | list[str]:
    """
    Manually prepends the bos token to the input.

    Args:
        bos_token (str): The BOS token to prepend.
        input (str | list[str]): The input to prepend the bos token to.

    Returns:
        str | list[str]: The input with the bos token manually prepended.
    """
    if isinstance(input, str):
        input = bos_token + input
    else:
        input = [bos_token + string for string in input]
    return input


def get_tokens_with_bos_removed(
    tokenizer: PreTrainedTokenizerBase, tokens: torch.Tensor
) -> torch.Tensor:
    """
    Removes the bos token from the beginning of each sequence in `tokens`.
    The last dimension of `tokens` must be the sequence length.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to tokenize the input.
        tokens (torch.Tensor): The tokenized input.

    Returns:
        torch.Tensor: The tokenized input with the bos token removed.
    """
    if tokenizer.padding_side == "right":
        return tokens[..., 1:]

    else:
        bos_removed_shape = list(tokens.shape)
        bos_removed_shape[-1] -= 1

        if tokenizer.bos_token_id == tokenizer.pad_token_id:
            is_not_pad_token = tokens.ne(tokenizer.pad_token_id)
            is_leading_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=False) == 0
            real_bos_positions = is_leading_pad.sum(-1) - 1
        else:
            real_bos_positions = (tokens == tokenizer.bos_token_id).int().argmax(-1)

        tokens = tokens.scatter(dim=1, index=real_bos_positions.unsqueeze(-1), value=-100)
        return tokens[tokens != -100].view(*bos_removed_shape)


def get_attention_mask(
    tokenizer: PreTrainedTokenizerBase, tokens: torch.Tensor, prepend_bos: bool
) -> torch.Tensor:
    """
    Computes the attention mask for the tokenized input.
    NOTE: Only the leftmost leading pads (when `padding_side == left`)
    or rightmost trailing pads (when `padding_side == right`) are
    considered as real pad tokens that should not be attended.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenization.
        tokens (torch.Tensor): The tokenized input.
        prepend_bos (bool): If True, a BOS token is prepended to the input.

    Returns:
        torch.Tensor: The attention mask for the input.
    """

    # Initialize the attention mask with ones (indicating all tokens should be attended to)
    attention_mask = torch.ones_like(tokens)
    if tokenizer is None:
        return attention_mask
    is_not_pad_token = tokens.ne(tokenizer.pad_token_id)

    if tokenizer.padding_side == "right":
        # Zero-out the rightmost trailing pad tokens
        is_trailing_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=True) == 0
        attention_mask[is_trailing_pad] = 0
    else:
        # Zero-out the leftmost leading pad tokens
        is_leading_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=False) == 0
        attention_mask[is_leading_pad] = 0

        # Unmask BOS when it shares the same ID as pad token
        if prepend_bos and tokenizer.bos_token_id == tokenizer.pad_token_id:
            pad_bos_positions = is_leading_pad.sum(-1) - 1
            attention_mask[torch.arange(attention_mask.shape[0]), pad_bos_positions] = 1

    return attention_mask
