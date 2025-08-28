"""Architecture adapter factory.

This module provides a factory for creating architecture adapters.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.supported_architectures import (
    BertArchitectureAdapter,
    BloomArchitectureAdapter,
    Gemma1ArchitectureAdapter,
    Gemma2ArchitectureAdapter,
    Gemma3ArchitectureAdapter,
    GPT2ArchitectureAdapter,
    Gpt2LmHeadCustomArchitectureAdapter,
    GptjArchitectureAdapter,
    GPTOSSArchitectureAdapter,
    LlamaArchitectureAdapter,
    MingptArchitectureAdapter,
    MistralArchitectureAdapter,
    MixtralArchitectureAdapter,
    NanogptArchitectureAdapter,
    NeelSoluOldArchitectureAdapter,
    NeoArchitectureAdapter,
    NeoxArchitectureAdapter,
    OptArchitectureAdapter,
    Phi3ArchitectureAdapter,
    PhiArchitectureAdapter,
    Qwen2ArchitectureAdapter,
    QwenArchitectureAdapter,
    T5ArchitectureAdapter,
)

# Export supported architectures
SUPPORTED_ARCHITECTURES = {
    "BertForMaskedLM": BertArchitectureAdapter,
    "BloomForCausalLM": BloomArchitectureAdapter,
    "GemmaForCausalLM": Gemma1ArchitectureAdapter,  # Default to Gemma1 as it's the original version
    "Gemma1ForCausalLM": Gemma1ArchitectureAdapter,
    "Gemma2ForCausalLM": Gemma2ArchitectureAdapter,
    "Gemma3ForCausalLM": Gemma3ArchitectureAdapter,
    "GPT2LMHeadModel": GPT2ArchitectureAdapter,
    "GptOssForCausalLM": GPTOSSArchitectureAdapter,
    "GPT2LMHeadCustomModel": Gpt2LmHeadCustomArchitectureAdapter,
    "GPTJForCausalLM": GptjArchitectureAdapter,
    "LlamaForCausalLM": LlamaArchitectureAdapter,
    "MixtralForCausalLM": MixtralArchitectureAdapter,
    "MistralForCausalLM": MistralArchitectureAdapter,
    "NeoForCausalLM": NeoArchitectureAdapter,
    "NeoXForCausalLM": NeoxArchitectureAdapter,
    "NeelSoluOldForCausalLM": NeelSoluOldArchitectureAdapter,
    "OPTForCausalLM": OptArchitectureAdapter,
    "PhiForCausalLM": PhiArchitectureAdapter,
    "Phi3ForCausalLM": Phi3ArchitectureAdapter,
    "QwenForCausalLM": QwenArchitectureAdapter,
    "Qwen2ForCausalLM": Qwen2ArchitectureAdapter,
    "T5ForConditionalGeneration": T5ArchitectureAdapter,
    "NanoGPTForCausalLM": NanogptArchitectureAdapter,
    "MinGPTForCausalLM": MingptArchitectureAdapter,
    "GPTNeoForCausalLM": NeoArchitectureAdapter,
    "GPTNeoXForCausalLM": NeoxArchitectureAdapter,
}


class ArchitectureAdapterFactory:
    """Factory for creating architecture adapters."""

    _adapters = SUPPORTED_ARCHITECTURES

    @classmethod
    def select_architecture_adapter(cls, cfg: Any) -> ArchitectureAdapter:
        """Select the appropriate architecture adapter for the given config (HF or TL).

        Args:
            cfg: The config to select the adapter for (can be Hugging Face or TL config).

        Returns:
            The selected architecture adapter.

        Raises:
            ValueError: If no adapter is found for the given config.
        """
        # Try to extract architecture name from Hugging Face config
        architectures = []
        if hasattr(cfg, "original_architecture"):
            architectures.append(cfg.original_architecture)
        if hasattr(cfg, "architectures") and cfg.architectures:
            architectures.extend(cfg.architectures)
        if hasattr(cfg, "model_type"):
            # Try to map model_type to a known architecture
            # e.g. 'gemma3' -> 'Gemma3ForCausalLM'
            model_type = cfg.model_type
            # Try to find a matching adapter by model_type
            for arch_name in cls._adapters:
                if model_type.lower() in arch_name.lower():
                    architectures.append(arch_name)

        # Try each architecture in order
        for architecture in architectures:
            if architecture in cls._adapters:
                return cls._adapters[architecture](cfg)

        # If no architecture was found, raise an error
        raise ValueError(f"Could not determine architecture from config: {cfg}")
