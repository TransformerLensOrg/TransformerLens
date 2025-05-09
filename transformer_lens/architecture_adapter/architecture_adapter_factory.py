"""Architecture adapter factory.

This module provides a factory for creating architecture adapters.
"""

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.architecture_adapter.supported_architectures import (
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
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class ArchitectureAdapterFactory:
    """Factory for creating architecture adapters."""

    _adapters: dict[str, type[ArchitectureConversion]] = {
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
    }

    @classmethod
    def select_architecture_adapter(cls, cfg: HookedTransformerConfig) -> ArchitectureConversion:
        """Select the appropriate architecture adapter for the given config.

        Args:
            cfg: The config to select the adapter for.

        Returns:
            The selected architecture adapter.

        Raises:
            ValueError: If no adapter is found for the given config.
        """
        architecture = cfg.original_architecture
        if architecture not in cls._adapters:
            raise ValueError(f"No adapter found for architecture: {architecture}")
        return cls._adapters[architecture](cfg) 