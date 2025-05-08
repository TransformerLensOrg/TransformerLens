"""Model bootstrapping utilities.

This module provides a streamlined way to load models using architecture adapters.
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from transformer_lens.architecture_adapter.bridge import TransformerBridge
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def get_base_config(hf_config: Any) -> dict[str, Any]:
    """Get the base config from a HuggingFace config.

    Args:
        hf_config: The HuggingFace config to get the base config from.

    Returns:
        The base config.
    """
    # Handle GPT-2 style configs
    if hasattr(hf_config, "n_embd"):
        base_config = {
            "n_layers": hf_config.n_layer,
            "d_model": hf_config.n_embd,
            "n_ctx": hf_config.n_positions,
            "n_heads": hf_config.n_head,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,  # Common for GPT-style models
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
        }
    # Handle other model configs
    else:
        base_config = {
            "n_layers": getattr(hf_config, "num_hidden_layers", None),
            "d_model": getattr(hf_config, "hidden_size", None),
            "n_ctx": getattr(hf_config, "max_position_embeddings", None),
            "n_heads": getattr(hf_config, "num_attention_heads", None),
            "d_head": None,  # Will be calculated below
            "d_mlp": None,  # Will be calculated below
            "d_vocab": getattr(hf_config, "vocab_size", None),
            "act_fn": getattr(hf_config, "hidden_act", "gelu"),
        }

        # Calculate d_head
        if base_config["d_model"] is not None and base_config["n_heads"] is not None:
            base_config["d_head"] = base_config["d_model"] // base_config["n_heads"]

        # Calculate d_mlp based on model architecture
        if hasattr(hf_config, "intermediate_size"):
            base_config["d_mlp"] = hf_config.intermediate_size
        elif hasattr(hf_config, "n_inner"):
            base_config["d_mlp"] = hf_config.n_inner

    # Validate required fields
    missing_fields = [k for k, v in base_config.items() if v is None]
    if missing_fields:
        raise ValueError(
            f"Could not determine the following required config fields: {missing_fields}"
        )

    return base_config


def boot(
    model_id: str,
    config_dict: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> TransformerBridge:
    """Load a model using architecture adapters.

    This function provides a streamlined way to load models using architecture adapters.
    It will:
    1. Load the model and tokenizer from HuggingFace
    2. Create a config from the model
    3. Select the appropriate architecture adapter
    4. Create a bridge to the model

    Args:
        model_id: The HuggingFace model ID to load
        config_dict: Optional dictionary of config parameters to override defaults
        device: The device to load the model onto
        dtype: The dtype to load the model in
        **kwargs: Additional arguments to pass to the model loading process

    Returns:
        TransformerBridge: The bridge to the loaded model
    """
    # Load the model and tokenizer from HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        **kwargs,
    )

    # Create config from model
    hf_config = hf_model.config
    base_config = {
        "model_name": model_id,
        "original_architecture": hf_config.architectures[0],
        **get_base_config(hf_config),
    }
    cfg = HookedTransformerConfig.from_dict({**base_config, **(config_dict or {})})

    # Select the appropriate architecture adapter
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)

    # Create and return the bridge
    return TransformerBridge(
        hf_model,
        adapter,
        device=device,
        dtype=dtype,
    ) 