"""Architecture adapter module.

This module provides functionality to adapt different transformer architectures to a common interface.
"""

from transformer_lens.architecture_adapter.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.architecture_adapter.bridge import TransformerBridge
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

__all__ = [
    "ArchitectureAdapterFactory",
    "ArchitectureConversion",
    "TransformerBridge",
    "BertArchitectureAdapter",
    "BloomArchitectureAdapter",
    "Gemma1ArchitectureAdapter",
    "Gemma2ArchitectureAdapter",
    "Gemma3ArchitectureAdapter",
    "GPT2ArchitectureAdapter",
    "GPT2LMHeadCustomArchitectureAdapter",
    "GPTJArchitectureAdapter",
    "LlamaArchitectureAdapter",
    "MinGPTArchitectureAdapter",
    "MistralArchitectureAdapter",
    "MixtralArchitectureAdapter",
    "NanoGPTArchitectureAdapter",
    "NeelSoluOldArchitectureAdapter",
    "NeoArchitectureAdapter",
    "NeoXArchitectureAdapter",
    "OPTArchitectureAdapter",
    "Phi3ArchitectureAdapter",
    "PhiArchitectureAdapter",
    "Qwen2ArchitectureAdapter",
    "QwenArchitectureAdapter",
    "T5ArchitectureAdapter",
]
