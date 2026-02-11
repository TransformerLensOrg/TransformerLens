"""Supported architecture adapters.

This module contains all the supported architecture adapters for different model architectures.
"""

from transformer_lens.model_bridge.supported_architectures.bert import (
    BertArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.bloom import (
    BloomArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma1 import (
    Gemma1ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma2 import (
    Gemma2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt_oss import GPTOSSArchitectureAdapter
from transformer_lens.model_bridge.supported_architectures.gpt2_lm_head_custom import (
    Gpt2LmHeadCustomArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gptj import (
    GptjArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mingpt import (
    MingptArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mixtral import (
    MixtralArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.nanogpt import (
    NanogptArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.neel_solu_old import (
    NeelSoluOldArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.neo import (
    NeoArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.neox import (
    NeoxArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.opt import (
    OptArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.phi import (
    PhiArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.pythia import (
    PythiaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen import (
    QwenArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.stablelm import (
    StableLmArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)

__all__ = [
    "BertArchitectureAdapter",
    "BloomArchitectureAdapter",
    "Gemma1ArchitectureAdapter",
    "Gemma2ArchitectureAdapter",
    "Gemma3ArchitectureAdapter",
    "GPT2ArchitectureAdapter",
    "Gpt2LmHeadCustomArchitectureAdapter",
    "GptjArchitectureAdapter",
    "LlamaArchitectureAdapter",
    "MingptArchitectureAdapter",
    "MistralArchitectureAdapter",
    "MixtralArchitectureAdapter",
    "NanogptArchitectureAdapter",
    "NeelSoluOldArchitectureAdapter",
    "NeoArchitectureAdapter",
    "NeoxArchitectureAdapter",
    "OptArchitectureAdapter",
    "PhiArchitectureAdapter",
    "Phi3ArchitectureAdapter",
    "PythiaArchitectureAdapter",
    "QwenArchitectureAdapter",
    "Qwen2ArchitectureAdapter",
    "Qwen3ArchitectureAdapter",
    "StableLmArchitectureAdapter",
    "T5ArchitectureAdapter",
]
