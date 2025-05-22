"""Architecture adapter factory.

This module provides a factory for creating architecture adapters.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.model_bridge.supported_architectures import (
    BertArchitectureAdapter,
    BloomArchitectureAdapter,
    Gemma1ArchitectureAdapter,
    Gemma2ArchitectureAdapter,
    Gemma3ArchitectureAdapter,
    GPT2ArchitectureAdapter,
    GPT2LMHeadCustomArchitectureAdapter,
    GPTJArchitectureAdapter,
    LlamaArchitectureAdapter,
    MinGPTArchitectureAdapter,
    MistralArchitectureAdapter,
    MixtralArchitectureAdapter,
    NanoGPTArchitectureAdapter,
    NeelSoluOldArchitectureAdapter,
    NeoArchitectureAdapter,
    NeoXArchitectureAdapter,
    OPTArchitectureAdapter,
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
    "GPT2LMHeadCustomModel": GPT2LMHeadCustomArchitectureAdapter,
    "GPTJForCausalLM": GPTJArchitectureAdapter,
    "LlamaForCausalLM": LlamaArchitectureAdapter,
    "MixtralForCausalLM": MixtralArchitectureAdapter,
    "MistralForCausalLM": MistralArchitectureAdapter,
    "NeoForCausalLM": NeoArchitectureAdapter,
    "NeoXForCausalLM": NeoXArchitectureAdapter,
    "NeelSoluOldForCausalLM": NeelSoluOldArchitectureAdapter,
    "OPTForCausalLM": OPTArchitectureAdapter,
    "PhiForCausalLM": PhiArchitectureAdapter,
    "Phi3ForCausalLM": Phi3ArchitectureAdapter,
    "QwenForCausalLM": QwenArchitectureAdapter,
    "Qwen2ForCausalLM": Qwen2ArchitectureAdapter,
    "T5ForConditionalGeneration": T5ArchitectureAdapter,
    "NanoGPTForCausalLM": NanoGPTArchitectureAdapter,
    "MinGPTForCausalLM": MinGPTArchitectureAdapter,
    "GPTNeoForCausalLM": NeoArchitectureAdapter,
    "GPTNeoXForCausalLM": NeoXArchitectureAdapter,
}

class ArchitectureAdapterFactory:
    """Factory for creating architecture adapters."""

    _adapters = SUPPORTED_ARCHITECTURES

    @classmethod
    def select_architecture_adapter(cls, cfg: Any) -> ArchitectureConversion:
        """Select the appropriate architecture adapter for the given config (HF or TL).

        Args:
            cfg: The config to select the adapter for (can be Hugging Face or TL config).

        Returns:
            The selected architecture adapter.

        Raises:
            ValueError: If no adapter is found for the given config.
        """
        # Try to extract architecture name from Hugging Face config
        architecture = None
        if hasattr(cfg, 'original_architecture'):
            architecture = cfg.original_architecture
        elif hasattr(cfg, 'architectures') and cfg.architectures:
            architecture = cfg.architectures[0]
        elif hasattr(cfg, 'model_type'):
            # Try to map model_type to a known architecture
            # e.g. 'gemma3' -> 'Gemma3ForCausalLM'
            model_type = cfg.model_type
            # Try to find a matching adapter by model_type
            for arch_name in cls._adapters:
                if model_type.lower() in arch_name.lower():
                    architecture = arch_name
                    break
        if not architecture:
            raise ValueError(f"Could not determine architecture from config: {cfg}")
        if architecture not in cls._adapters:
            raise ValueError(f"No adapter found for architecture: {architecture}")
        return cls._adapters[architecture](cfg) 