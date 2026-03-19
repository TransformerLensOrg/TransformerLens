"""Model Registry tools for TransformerLens.

This package provides tools for discovering and documenting HuggingFace models
that are compatible with TransformerLens.

Main modules:
    - api: Public API for programmatic access to model registry data
    - schemas: Data classes for model entries, architecture gaps, etc.
    - verification: Verification tracking for model compatibility
    - exceptions: Custom exceptions for the model registry

Example usage:
    >>> from transformer_lens.tools.model_registry import api  # doctest: +SKIP
    >>> api.is_model_supported("openai-community/gpt2")  # doctest: +SKIP
    True
    >>> models = api.get_architecture_models("GPT2LMHeadModel")  # doctest: +SKIP
"""

from .exceptions import (
    ArchitectureNotSupportedError,
    DataNotLoadedError,
    DataValidationError,
    ModelNotFoundError,
    ModelRegistryError,
)
from .schemas import (
    ArchitectureAnalysis,
    ArchitectureGap,
    ArchitectureGapsReport,
    ArchitectureStats,
    ModelEntry,
    ModelMetadata,
    ScanInfo,
    SupportedModelsReport,
)
from .verification import VerificationHistory, VerificationRecord

# Canonical set of HuggingFace architecture class names supported by TransformerBridge.
# These must match the exact strings found in HF model config.architectures[]
# and correspond to adapters registered in architecture_adapter_factory.py.
#
# Internal-only architectures (NanoGPT, MinGPT, NeelSoluOld, GPT2LMHeadCustomModel)
# are excluded since they never appear on HuggingFace Hub.
HF_SUPPORTED_ARCHITECTURES: set[str] = {
    "ApertusForCausalLM",
    "BertForMaskedLM",
    "BloomForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "GraniteForCausalLM",
    "GraniteMoeForCausalLM",
    "GraniteMoeHybridForCausalLM",
    "GPT2LMHeadModel",
    "GptOssForCausalLM",
    "GPTJForCausalLM",
    "GPTNeoForCausalLM",
    "OpenELMForCausalLM",
    "GPTNeoXForCausalLM",
    "HubertForCTC",
    "HubertModel",
    "LlamaForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Olmo2ForCausalLM",
    "Olmo3ForCausalLM",
    "OlmoForCausalLM",
    "OlmoeForCausalLM",
    "OPTForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "QwenForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "StableLmForCausalLM",
    "T5ForConditionalGeneration",
}

__all__ = [
    # Constants
    "HF_SUPPORTED_ARCHITECTURES",
    # Exceptions
    "ModelRegistryError",
    "ModelNotFoundError",
    "ArchitectureNotSupportedError",
    "DataNotLoadedError",
    "DataValidationError",
    # Schemas
    "ModelEntry",
    "ModelMetadata",
    "ScanInfo",
    "ArchitectureGap",
    "ArchitectureStats",
    "ArchitectureAnalysis",
    "SupportedModelsReport",
    "ArchitectureGapsReport",
    # Verification
    "VerificationRecord",
    "VerificationHistory",
]
