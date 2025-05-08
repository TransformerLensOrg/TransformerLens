"""Boot module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from transformer_lens.architecture_adapter.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.architecture_adapter.bridge import TransformerBridge
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens.utilities.devices import get_device_for_block_index


def boot(
    model_name: str,
    config: HookedTransformerConfig | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        config: The config to use. If None, will be loaded from HuggingFace.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.
        **kwargs: Additional arguments to pass to the config.

    Returns:
        TransformerBridge: The bridge to the loaded model.
    """
    if config is None:
        config = get_pretrained_model_config(model_name, **kwargs)

    if device is None:
        device = get_device_for_block_index(0, config)

    # Load the model from HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        **kwargs,
    )

    # Create and return the bridge
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(config)
    return TransformerBridge(
        hf_model,
        adapter,
        device=device,
        dtype=dtype,
    ) 