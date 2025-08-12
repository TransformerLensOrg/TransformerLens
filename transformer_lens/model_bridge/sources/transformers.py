"""Transformers module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""


import copy
import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.utils import get_tokenizer_with_bos


def map_default_transformer_lens_config(hf_config):
    """Map HuggingFace config fields to TransformerLens config format.

    This function provides a standardized mapping from various HuggingFace config
    field names to the consistent TransformerLens naming convention.

    Args:
        hf_config: The HuggingFace config object

    Returns:
        A copy of hf_config with additional TransformerLens fields
    """
    tl_config = copy.deepcopy(hf_config)

    # Map d_model (hidden dimension)
    if hasattr(hf_config, "n_embd"):  # GPT2-style
        tl_config.d_model = hf_config.n_embd
    elif hasattr(hf_config, "hidden_size"):  # LLaMA/BERT-style
        tl_config.d_model = hf_config.hidden_size
    elif hasattr(hf_config, "d_model"):  # T5-style
        tl_config.d_model = hf_config.d_model

    # Map number of attention heads
    if hasattr(hf_config, "n_head"):  # GPT2-style
        tl_config.n_heads = hf_config.n_head
    elif hasattr(hf_config, "num_attention_heads"):  # LLaMA/BERT-style
        tl_config.n_heads = hf_config.num_attention_heads
    elif hasattr(hf_config, "num_heads"):  # T5-style
        tl_config.n_heads = hf_config.num_heads

    # Map number of layers
    if hasattr(hf_config, "n_layer"):  # GPT2-style
        tl_config.n_layers = hf_config.n_layer
    elif hasattr(hf_config, "num_hidden_layers"):  # LLaMA/BERT-style
        tl_config.n_layers = hf_config.num_hidden_layers
    elif hasattr(hf_config, "num_layers"):  # T5-style
        tl_config.n_layers = hf_config.num_layers

    # Map vocabulary size
    if hasattr(hf_config, "vocab_size"):
        tl_config.d_vocab = hf_config.vocab_size

    # Map context length
    if hasattr(hf_config, "n_positions"):  # GPT2-style
        tl_config.n_ctx = hf_config.n_positions
    elif hasattr(hf_config, "max_position_embeddings"):  # LLaMA/BERT-style
        tl_config.n_ctx = hf_config.max_position_embeddings
    elif hasattr(hf_config, "max_length"):  # Some models
        tl_config.n_ctx = hf_config.max_length

    # Map MLP dimension
    if hasattr(hf_config, "n_inner"):  # GPT2 explicit
        tl_config.d_mlp = hf_config.n_inner
    elif hasattr(hf_config, "intermediate_size"):  # BERT/LLaMA-style
        tl_config.d_mlp = hf_config.intermediate_size
    elif hasattr(tl_config, "d_model"):  # Default to 4x for GPT2-style
        tl_config.d_mlp = getattr(hf_config, "n_inner", 4 * tl_config.d_model)

    # Calculate d_head if we have both d_model and n_heads
    if hasattr(tl_config, "d_model") and hasattr(tl_config, "n_heads"):
        tl_config.d_head = tl_config.d_model // tl_config.n_heads

    # Set common defaults for transformer models
    tl_config.default_prepend_bos = True

    return tl_config


def boot(
    model_name: str,
    hf_config_overrides: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.
        tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.

    Returns:
        The bridge to the loaded model.
    """
    # Lazy import to avoid circular import
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    hf_config = AutoConfig.from_pretrained(model_name, output_attentions=True)

    # Apply config variables to hf_config before selecting adapter
    if hf_config_overrides:
        hf_config.__dict__.update(hf_config_overrides)

    # Apply HuggingFace to TransformerLens config mapping
    tl_config = map_default_transformer_lens_config(hf_config)

    adapter = ArchitectureAdapterFactory.select_architecture_adapter(tl_config)

    # Add device information to the config
    if device is not None:
        adapter.cfg.device = device

    # Load the model from HuggingFace using the original config
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=hf_config,
        torch_dtype=dtype,
    )

    # Move model to device if specified
    if device is not None:
        hf_model = hf_model.to(device)

    # Load the tokenizer
    tokenizer = tokenizer
    default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
    add_bos_token = getattr(adapter.cfg, "add_bos_token", True)

    if tokenizer is not None:
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
    else:
        huggingface_token = os.environ.get("HF_TOKEN", "")
        tokenizer = setup_tokenizer(
            AutoTokenizer.from_pretrained(
                model_name,
                add_bos_token=add_bos_token,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            ),
            default_padding_side=default_padding_side,
        )

    # Set tokenizer_prepends_bos configuration
    if tokenizer is not None:
        adapter.cfg.tokenizer_prepends_bos = len(tokenizer.encode("")) > 0

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


setattr(TransformerBridge, "boot_transformers", staticmethod(boot))
