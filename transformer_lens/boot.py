"""Boot module for TransformerLens.

This module provides functionality to load and convert models from HuggingFace to TransformerLens format.
"""


import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge import ArchitectureAdapterFactory
from transformer_lens.model_bridge.bridge import TransformerBridge


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

    # Load the model from HuggingFace using the original config
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=hf_config,
        torch_dtype=dtype,
        **merged_config,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    return TransformerBridge(
        hf_model,
        adapter,
        tokenizer,
    )
