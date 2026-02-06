"""Architecture adapter factory.

This module provides a factory for creating architecture adapters.
"""

from transformer_lens.config import TransformerBridgeConfig
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
    Qwen3ArchitectureAdapter,
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
    "Qwen3ForCausalLM": Qwen3ArchitectureAdapter,
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
    def select_architecture_adapter(cls, cfg: TransformerBridgeConfig) -> ArchitectureAdapter:
        """Select the appropriate architecture adapter for the given config.

        Args:
            cfg: The TransformerBridgeConfig to select the adapter for.

        Returns:
            The selected architecture adapter.

        Raises:
            ValueError: If no adapter is found for the given config.
        """
        if cfg.architecture is not None:
            if cfg.architecture in cls._adapters:
                return cls._adapters[cfg.architecture](cfg)
            else:
                raise ValueError(f"Unsupported architecture: {cfg.architecture}")

        # If architecture is None, this is an error since TransformerBridgeConfig should always have it set
        raise ValueError(f"TransformerBridgeConfig must have architecture set, got: {cfg}")
