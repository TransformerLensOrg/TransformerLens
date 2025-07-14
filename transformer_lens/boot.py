"""Boot module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""


import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from transformer_lens.model_bridge import ArchitectureAdapterFactory
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.utils import get_tokenizer_with_bos


def boot(
    model_name: str,
    config: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        config: The config dict to use. If None, will be loaded from HuggingFace.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.
        **kwargs: Additional keyword arguments for from_pretrained.

    Returns:
        The bridge to the loaded model.
    """
    hf_config = AutoConfig.from_pretrained(model_name, **kwargs)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(hf_config)
    default_config = adapter.default_cfg
    merged_config = {**default_config, **(config or {})}

    # Ensure d_mlp is set if intermediate_size is present
    if "d_mlp" not in merged_config and "intermediate_size" in merged_config:
        merged_config["d_mlp"] = merged_config["intermediate_size"]

    # Load the model from HuggingFace using the original config
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=hf_config,
        torch_dtype=dtype,
        **merged_config,
    )

    # Load the tokenizer
    tokenizer = kwargs.get("tokenizer", None)
    default_padding_side = kwargs.get("default_padding_side", None)

    if tokenizer is not None:
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        tokenizer = setup_tokenizer(
            AutoTokenizer.from_pretrained(
                model_name,
                add_bos_token=True,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            ),
            default_padding_side=default_padding_side,
        )

    return TransformerBridge(
        hf_model,
        adapter,
        tokenizer,
    )


def setup_tokenizer(
    tokenizer,
    default_padding_side=None,
):
    """Set's up the tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
        default_padding_side (str): "right" or "left", which side to pad on.

    """
    assert isinstance(
        tokenizer, PreTrainedTokenizerBase
    ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"

    assert default_padding_side in [
        "right",
        "left",
        None,
    ], f"padding_side must be 'right', 'left' or 'None', got {default_padding_side}"

    # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
    # Such a tokenizer should be set as the default tokenizer because the tokenization of some
    # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
    # prepended, and add_bos_token cannot be dynamically controlled after initialization
    # (https://github.com/huggingface/transformers/issues/25886).
    tokenizer_with_bos = get_tokenizer_with_bos(tokenizer)
    tokenizer = tokenizer_with_bos
    assert tokenizer is not None  # keep mypy happy

    # If user passes default_padding_side explicitly, use that value
    if default_padding_side is not None:
        tokenizer.padding_side = default_padding_side
    # If not, then use the tokenizer's default padding side
    # If the tokenizer doesn't have a default padding side, use the global default "right"
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    return tokenizer
