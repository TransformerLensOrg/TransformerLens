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
# Internal-only architectures (NanoGPT, MinGPT, NeelSoluOld, GPT2LMHeadCustomModel,
# TransformerLensNative) are excluded since they never appear on HuggingFace Hub.
# Factory-internal alias casings (Gemma1, Neo, NeoX) are also excluded since they
# route to canonical adapters but HF reports the canonical names (Gemma, GPTNeo,
# GPTNeoX) in config.architectures instead.
HF_SUPPORTED_ARCHITECTURES: set[str] = {
    "ApertusForCausalLM",
    "BaiChuanForCausalLM",
    "BaichuanForCausalLM",
    "BertForMaskedLM",
    "BloomForCausalLM",
    "CodeGenForCausalLM",
    "CohereForCausalLM",
    "DeepseekV3ForCausalLM",
    "FalconForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForConditionalGeneration",
    "GraniteForCausalLM",
    "GraniteMoeForCausalLM",
    "GraniteMoeHybridForCausalLM",
    "GPT2LMHeadModel",
    "GPTBigCodeForCausalLM",
    "GptOssForCausalLM",
    "GPTJForCausalLM",
    "GPTNeoForCausalLM",
    "OpenELMForCausalLM",
    "GPTNeoXForCausalLM",
    "HubertForCTC",
    "HubertModel",
    "InternLM2ForCausalLM",
    "LlamaForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MambaForCausalLM",
    "Mamba2ForCausalLM",
    "MPTForCausalLM",
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
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "SmolLM3ForCausalLM",
    "StableLmForCausalLM",
    "T5ForConditionalGeneration",
    "MT5ForConditionalGeneration",
    "XGLMForCausalLM",
}

# Foundation-trained orgs per architecture. Source of truth for the scraper's
# download-threshold bypass and the docs table's "Canonical only" toggle.
CANONICAL_AUTHORS_BY_ARCH: dict[str, list[str]] = {
    "ApertusForCausalLM": ["swiss-ai"],
    "BaiChuanForCausalLM": ["baichuan-inc"],
    "BaichuanForCausalLM": ["baichuan-inc"],
    "BertForMaskedLM": ["google-bert"],
    "BloomForCausalLM": ["bigscience"],
    "CodeGenForCausalLM": ["Salesforce"],
    "CohereForCausalLM": ["CohereLabs"],
    "DeepseekV3ForCausalLM": ["deepseek-ai"],
    "FalconForCausalLM": ["tiiuae"],
    "Gemma2ForCausalLM": ["google"],
    "Gemma3ForCausalLM": ["google"],
    "Gemma3ForConditionalGeneration": ["google"],
    "Gemma3nForConditionalGeneration": ["google"],
    "GemmaForCausalLM": ["google"],
    "GPT2LMHeadModel": ["openai-community", "stanford-crfm", "Writer"],
    "GPTBigCodeForCausalLM": ["bigcode"],
    "GptOssForCausalLM": ["openai"],
    "GPTJForCausalLM": ["EleutherAI", "togethercomputer"],
    "GPTNeoForCausalLM": ["EleutherAI", "roneneldan"],
    "GPTNeoXForCausalLM": ["EleutherAI", "cyberagent", "stabilityai", "togethercomputer"],
    "GraniteForCausalLM": ["ibm-granite"],
    "GraniteMoeForCausalLM": ["ibm-granite"],
    "GraniteMoeHybridForCausalLM": ["ibm-granite"],
    "HubertForCTC": ["facebook"],
    "HubertModel": ["facebook"],
    "InternLM2ForCausalLM": ["internlm"],
    "LlamaForCausalLM": ["meta-llama", "huggyllama", "codellama", "SimpleStories"],
    "LlavaForConditionalGeneration": ["llava-hf"],
    "LlavaNextForConditionalGeneration": ["llava-hf"],
    "LlavaOnevisionForConditionalGeneration": ["llava-hf"],
    "Mamba2ForCausalLM": ["state-spaces"],
    "MambaForCausalLM": ["state-spaces"],
    "MistralForCausalLM": ["mistralai"],
    "MixtralForCausalLM": ["mistralai"],
    "MPTForCausalLM": ["mosaicml"],
    "MT5ForConditionalGeneration": ["google", "bigscience", "csebuetnlp"],
    "Olmo2ForCausalLM": ["allenai", "HPLT"],
    "Olmo3ForCausalLM": ["allenai"],
    "OlmoeForCausalLM": ["allenai"],
    "OlmoForCausalLM": ["allenai"],
    "OpenELMForCausalLM": ["apple"],
    "OPTForCausalLM": ["facebook"],
    "Phi3ForCausalLM": ["microsoft"],
    "PhiForCausalLM": ["microsoft"],
    "Qwen2ForCausalLM": ["Qwen", "nvidia"],
    "Qwen3ForCausalLM": ["Qwen", "nvidia"],
    "Qwen3MoeForCausalLM": ["Qwen"],
    "Qwen3NextForCausalLM": ["Qwen"],
    "Qwen3_5ForCausalLM": ["Qwen"],
    "Qwen3_5ForConditionalGeneration": ["Qwen"],
    "QwenForCausalLM": ["Qwen"],
    "SmolLM3ForCausalLM": ["HuggingFaceTB"],
    "StableLmForCausalLM": ["stabilityai"],
    "T5ForConditionalGeneration": ["google-t5", "google", "Salesforce", "MBZUAI"],
    "XGLMForCausalLM": ["facebook"],
}

__all__ = [
    # Constants
    "HF_SUPPORTED_ARCHITECTURES",
    "CANONICAL_AUTHORS_BY_ARCH",
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
