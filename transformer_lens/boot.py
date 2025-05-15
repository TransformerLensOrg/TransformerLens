"""Boot module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""

from typing import Any

import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

from transformer_lens.architecture_adapter.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.architecture_adapter.bridge import TransformerBridge
from transformer_lens.utilities.devices import get_device_for_block_index


def boot(
    model_name: str,
    config: Any = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> tuple[TransformerBridge, Any]:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        config: The config to use. If None, will be loaded from HuggingFace.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.
        **kwargs: Additional arguments to pass to the config.

    Returns:
        (TransformerBridge, tokenizer): The bridge to the loaded model and the tokenizer.
    """
    if config is None:
        # Download the config from Hugging Face
        config = AutoConfig.from_pretrained(model_name, **kwargs)

    # Try to get device if possible, fallback to 'cpu' if not available
    if device is None:
        try:
            device = get_device_for_block_index(0, config)
        except Exception:
            device = 'cpu'

    # Load the model from HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        **kwargs,
    )

    # Load the tokenizer
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create and return the bridge
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(config)
    return TransformerBridge(
        hf_model,
        adapter,
        tokenizer,
    ), tokenizer 