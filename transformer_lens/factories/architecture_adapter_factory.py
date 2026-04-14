"""Architecture adapter factory.

This module provides a factory for creating architecture adapters.
"""

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.supported_architectures import (
    ApertusArchitectureAdapter,
    BertArchitectureAdapter,
    BloomArchitectureAdapter,
    CodeGenArchitectureAdapter,
    CohereArchitectureAdapter,
    FalconArchitectureAdapter,
    Gemma1ArchitectureAdapter,
    Gemma2ArchitectureAdapter,
    Gemma3ArchitectureAdapter,
    Gemma3MultimodalArchitectureAdapter,
    GPT2ArchitectureAdapter,
    Gpt2LmHeadCustomArchitectureAdapter,
    GPTBigCodeArchitectureAdapter,
    GptjArchitectureAdapter,
    GPTOSSArchitectureAdapter,
    GraniteArchitectureAdapter,
    GraniteMoeArchitectureAdapter,
    GraniteMoeHybridArchitectureAdapter,
    HubertArchitectureAdapter,
    InternLM2ArchitectureAdapter,
    LlamaArchitectureAdapter,
    LlavaArchitectureAdapter,
    LlavaNextArchitectureAdapter,
    LlavaOnevisionArchitectureAdapter,
    Mamba2ArchitectureAdapter,
    MambaArchitectureAdapter,
    MingptArchitectureAdapter,
    MistralArchitectureAdapter,
    MixtralArchitectureAdapter,
    NanogptArchitectureAdapter,
    NeelSoluOldArchitectureAdapter,
    NeoArchitectureAdapter,
    NeoxArchitectureAdapter,
    Olmo2ArchitectureAdapter,
    Olmo3ArchitectureAdapter,
    OlmoArchitectureAdapter,
    OlmoeArchitectureAdapter,
    OpenElmArchitectureAdapter,
    OptArchitectureAdapter,
    Phi3ArchitectureAdapter,
    PhiArchitectureAdapter,
    Qwen2ArchitectureAdapter,
    Qwen3ArchitectureAdapter,
    Qwen3MoeArchitectureAdapter,
    Qwen3NextArchitectureAdapter,
    QwenArchitectureAdapter,
    StableLmArchitectureAdapter,
    T5ArchitectureAdapter,
    XGLMArchitectureAdapter,
)

# Export supported architectures
SUPPORTED_ARCHITECTURES = {
    "ApertusForCausalLM": ApertusArchitectureAdapter,
    "BertForMaskedLM": BertArchitectureAdapter,
    "BloomForCausalLM": BloomArchitectureAdapter,
    "CodeGenForCausalLM": CodeGenArchitectureAdapter,
    "CohereForCausalLM": CohereArchitectureAdapter,
    "FalconForCausalLM": FalconArchitectureAdapter,
    "GemmaForCausalLM": Gemma1ArchitectureAdapter,  # Default to Gemma1 as it's the original version
    "Gemma1ForCausalLM": Gemma1ArchitectureAdapter,
    "Gemma2ForCausalLM": Gemma2ArchitectureAdapter,
    "Gemma3ForCausalLM": Gemma3ArchitectureAdapter,
    "Gemma3ForConditionalGeneration": Gemma3MultimodalArchitectureAdapter,
    "GraniteForCausalLM": GraniteArchitectureAdapter,
    "GraniteMoeForCausalLM": GraniteMoeArchitectureAdapter,
    "GraniteMoeHybridForCausalLM": GraniteMoeHybridArchitectureAdapter,
    "GPT2LMHeadModel": GPT2ArchitectureAdapter,
    "GPTBigCodeForCausalLM": GPTBigCodeArchitectureAdapter,
    "GptOssForCausalLM": GPTOSSArchitectureAdapter,
    "GPT2LMHeadCustomModel": Gpt2LmHeadCustomArchitectureAdapter,
    "GPTJForCausalLM": GptjArchitectureAdapter,
    "HubertForCTC": HubertArchitectureAdapter,
    "HubertModel": HubertArchitectureAdapter,
    "InternLM2ForCausalLM": InternLM2ArchitectureAdapter,
    "LlamaForCausalLM": LlamaArchitectureAdapter,
    "LlavaForConditionalGeneration": LlavaArchitectureAdapter,
    "LlavaNextForConditionalGeneration": LlavaNextArchitectureAdapter,
    "LlavaOnevisionForConditionalGeneration": LlavaOnevisionArchitectureAdapter,
    "Mamba2ForCausalLM": Mamba2ArchitectureAdapter,
    "MambaForCausalLM": MambaArchitectureAdapter,
    "MixtralForCausalLM": MixtralArchitectureAdapter,
    "MistralForCausalLM": MistralArchitectureAdapter,
    "NeoForCausalLM": NeoArchitectureAdapter,
    "NeoXForCausalLM": NeoxArchitectureAdapter,
    "NeelSoluOldForCausalLM": NeelSoluOldArchitectureAdapter,
    "OlmoForCausalLM": OlmoArchitectureAdapter,
    "Olmo2ForCausalLM": Olmo2ArchitectureAdapter,
    "Olmo3ForCausalLM": Olmo3ArchitectureAdapter,
    "OlmoeForCausalLM": OlmoeArchitectureAdapter,
    "OpenELMForCausalLM": OpenElmArchitectureAdapter,
    "OPTForCausalLM": OptArchitectureAdapter,
    "PhiForCausalLM": PhiArchitectureAdapter,
    "Phi3ForCausalLM": Phi3ArchitectureAdapter,
    "QwenForCausalLM": QwenArchitectureAdapter,
    "Qwen2ForCausalLM": Qwen2ArchitectureAdapter,
    "Qwen3ForCausalLM": Qwen3ArchitectureAdapter,
    "Qwen3MoeForCausalLM": Qwen3MoeArchitectureAdapter,
    "Qwen3NextForCausalLM": Qwen3NextArchitectureAdapter,
    "StableLmForCausalLM": StableLmArchitectureAdapter,
    "T5ForConditionalGeneration": T5ArchitectureAdapter,
    "XGLMForCausalLM": XGLMArchitectureAdapter,
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
