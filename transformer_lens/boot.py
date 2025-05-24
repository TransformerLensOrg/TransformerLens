"""Boot module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""

import os
from typing import Any, Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.model_bridge import ArchitectureAdapterFactory
from transformer_lens.model_bridge.bridge import TransformerBridge


def boot(
    model_name: str,
    config: dict | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> TransformerBridge:
    """Boot a model from HuggingFace.

    Args:
        model_name: The name of the model to load.
        config: The config dict to use. If None, will be loaded from HuggingFace.
        device: The device to use. If None, will be determined automatically.
        dtype: The dtype to use for the model.

    Returns:
        The bridge to the loaded model.
    """
    hf_config = AutoConfig.from_pretrained(model_name)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(hf_config)
    default_config = adapter.default_config
    merged_config = {**default_config, **(config or {})}

    # Load the model from HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        **merged_config,
    )

    # Load the tokenizer (not returned)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return TransformerBridge(
        hf_model,
        adapter,
        tokenizer,
    ) 