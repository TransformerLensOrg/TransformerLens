"""Architecture adapter module.

This module contains adapters for converting between different model architectures.
"""

from transformer_lens.architecture_adapter.bert import BertArchitectureAdapter
from transformer_lens.architecture_adapter.bloom import BloomArchitectureAdapter
from transformer_lens.architecture_adapter.gemma import GemmaArchitectureAdapter
from transformer_lens.architecture_adapter.gpt2 import GPT2ArchitectureAdapter
from transformer_lens.architecture_adapter.gpt2_lm_head_custom import (
    GPT2LMHeadCustomArchitectureAdapter,
)
from transformer_lens.architecture_adapter.gptj import GPTJArchitectureAdapter
from transformer_lens.architecture_adapter.llama import LLAMAArchitectureAdapter
from transformer_lens.architecture_adapter.mistral import MistralArchitectureAdapter
from transformer_lens.architecture_adapter.mixtral import MixtralArchitectureAdapter
from transformer_lens.architecture_adapter.neo import NEOArchitectureAdapter
from transformer_lens.architecture_adapter.neox import NEOXArchitectureAdapter
from transformer_lens.architecture_adapter.phi import PhiArchitectureAdapter
from transformer_lens.architecture_adapter.qwen import QwenArchitectureAdapter
from transformer_lens.architecture_adapter.qwen2 import Qwen2ArchitectureAdapter
from transformer_lens.architecture_adapter.t5 import T5ArchitectureAdapter

__all__ = [
    "BertArchitectureAdapter",
    "BloomArchitectureAdapter",
    "GemmaArchitectureAdapter",
    "GPT2ArchitectureAdapter",
    "GPT2LMHeadCustomArchitectureAdapter",
    "GPTJArchitectureAdapter",
    "LLAMAArchitectureAdapter",
    "MistralArchitectureAdapter",
    "MixtralArchitectureAdapter",
    "NEOArchitectureAdapter",
    "NEOXArchitectureAdapter",
    "PhiArchitectureAdapter",
    "Qwen2ArchitectureAdapter",
    "QwenArchitectureAdapter",
    "T5ArchitectureAdapter",
]
