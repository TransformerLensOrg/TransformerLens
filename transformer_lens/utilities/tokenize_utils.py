"""tokenize_utils.

This module contains utility functions related to tokenization
"""

from __future__ import annotations

import os
from copy import deepcopy

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
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (PreTrainedTokenizerBase): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        if not hasattr(tokenizer, "eos_token") or tokenizer.eos_token is None:
            raise ValueError("Tokenizer must have an eos_token")
        full_text = tokenizer.eos_token.join(text)

        # Handle the case when full_text is empty
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        # Handle cases where num_tokens is less than seq_len
        if num_tokens < seq_len:
            num_batches = 1
            # Pad tokens if necessary
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                padding_length = seq_len - len(tokens)
                padding = np.full(padding_length, tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            num_batches = num_tokens // seq_len
            # Drop the final tokens if not enough to make a full sequence
            tokens = tokens[: seq_len * num_batches]

        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
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

        # If the bos token is the same as the pad token,
        # the last token of the leftmost leading pad tokens is the bos token.
        # We need to set the attention mask for the bos token to 1.
        if prepend_bos and tokenizer.bos_token_id == tokenizer.pad_token_id:
            pad_bos_positions = is_leading_pad.sum(-1) - 1
            attention_mask[torch.arange(attention_mask.shape[0]), pad_bos_positions] = 1

    return attention_mask
