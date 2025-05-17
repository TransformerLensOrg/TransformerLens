"""Supported architecture adapters.

This module contains all the supported architecture adapters for different model architectures.
"""

from transformer_lens.architecture_adapter.supported_architectures.bert import (
    BertArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.bloom import (
    BloomArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.gemma1 import (
    Gemma1ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.gemma2 import (
    Gemma2ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.gpt2_lm_head_custom import (
    GPT2LMHeadCustomArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.gptj import (
    GPTJArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.mingpt import (
    MinGPTArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.mixtral import (
    MixtralArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.nanogpt import (
    NanoGPTArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.neel_solu_old import (
    NeelSoluOldArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.neo import (
    NeoArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.neox import (
    NeoXArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.opt import (
    OPTArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.phi import (
    PhiArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.pythia import (
    PythiaArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.qwen import (
    QwenArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)

__all__ = [
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
    "PhiArchitectureAdapter",
    "Phi3ArchitectureAdapter",
    "PythiaArchitectureAdapter",
    "QwenArchitectureAdapter",
    "Qwen2ArchitectureAdapter",
    "T5ArchitectureAdapter",
] 