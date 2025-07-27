"""Transformers module for TransformerLens.

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

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.utils import get_tokenizer_with_bos


def map_default_transformer_lens_config(adapter, hf_config):
    """Map HuggingFace config fields to TransformerLens config format.

    This function provides a standardized mapping from various HuggingFace config
    field names to the consistent TransformerLens naming convention.

    Args:
        adapter: The architecture adapter
        hf_config: The HuggingFace config object
    """
    # Map d_model (hidden dimension)
    if hasattr(hf_config, "n_embd"):  # GPT2-style
        adapter.cfg.d_model = hf_config.n_embd
    elif hasattr(hf_config, "hidden_size"):  # LLaMA/BERT-style
        adapter.cfg.d_model = hf_config.hidden_size
    elif hasattr(hf_config, "d_model"):  # T5-style
        adapter.cfg.d_model = hf_config.d_model

    # Map number of attention heads
    if hasattr(hf_config, "n_head"):  # GPT2-style
        adapter.cfg.n_heads = hf_config.n_head
    elif hasattr(hf_config, "num_attention_heads"):  # LLaMA/BERT-style
        adapter.cfg.n_heads = hf_config.num_attention_heads
    elif hasattr(hf_config, "num_heads"):  # T5-style
        adapter.cfg.n_heads = hf_config.num_heads

    # Map number of layers
    if hasattr(hf_config, "n_layer"):  # GPT2-style
        adapter.cfg.n_layers = hf_config.n_layer
    elif hasattr(hf_config, "num_hidden_layers"):  # LLaMA/BERT-style
        adapter.cfg.n_layers = hf_config.num_hidden_layers
    elif hasattr(hf_config, "num_layers"):  # T5-style
        adapter.cfg.n_layers = hf_config.num_layers

    # Map vocabulary size
    if hasattr(hf_config, "vocab_size"):
        adapter.cfg.d_vocab = hf_config.vocab_size

    # Map context length
    if hasattr(hf_config, "n_positions"):  # GPT2-style
        adapter.cfg.n_ctx = hf_config.n_positions
    elif hasattr(hf_config, "max_position_embeddings"):  # LLaMA/BERT-style
        adapter.cfg.n_ctx = hf_config.max_position_embeddings
    elif hasattr(hf_config, "max_length"):  # Some models
        adapter.cfg.n_ctx = hf_config.max_length

    # Map MLP dimension
    if hasattr(hf_config, "n_inner"):  # GPT2 explicit
        adapter.cfg.d_mlp = hf_config.n_inner
    elif hasattr(hf_config, "intermediate_size"):  # BERT/LLaMA-style
        adapter.cfg.d_mlp = hf_config.intermediate_size
    elif hasattr(adapter.cfg, "d_model"):  # Default to 4x for GPT2-style
        adapter.cfg.d_mlp = getattr(hf_config, "n_inner", 4 * adapter.cfg.d_model)

    # Calculate d_head if we have both d_model and n_heads
    if hasattr(adapter.cfg, "d_model") and hasattr(adapter.cfg, "n_heads"):
        adapter.cfg.d_head = adapter.cfg.d_model // adapter.cfg.n_heads

    # Set common defaults for transformer models
    if not hasattr(adapter.cfg, "default_prepend_bos"):
        adapter.cfg.default_prepend_bos = True


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
    # Lazy import to avoid circular import
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    hf_config = AutoConfig.from_pretrained(model_name, **kwargs)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(hf_config)

    # Get default config from adapter and merge with passed config
    default_config = adapter.default_cfg
    merged_config = {**default_config, **(config or {})}

    # Ensure d_mlp is set if intermediate_size is present
    if "d_mlp" not in merged_config and "intermediate_size" in merged_config:
        merged_config["d_mlp"] = merged_config["intermediate_size"]

    # Apply merged config to adapter
    for key, value in merged_config.items():
        setattr(adapter.cfg, key, value)

    # Apply HuggingFace to TransformerLens config mapping
    map_default_transformer_lens_config(adapter, hf_config)

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
