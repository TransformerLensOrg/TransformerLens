"""Supported architecture adapters.

This module contains all the supported architecture adapters for different model architectures.
"""

from transformer_lens.model_bridge.supported_architectures.apertus import (
    ApertusArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.bert import (
    BertArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.bloom import (
    BloomArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.codegen import (
    CodeGenArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.falcon import (
    FalconArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.gemma3_multimodal import (
    Gemma3MultimodalArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt2_lm_head_custom import (
    Gpt2LmHeadCustomArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt_oss import (
    GPTOSSArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gptj import (
    GptjArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.granite import (
    GraniteArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.granite_moe import (
    GraniteMoeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.granite_moe_hybrid import (
    GraniteMoeHybridArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.hubert import (
    HubertArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llava_next import (
    LlavaNextArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llava_onevision import (
    LlavaOnevisionArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.olmo import (
    OlmoArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.olmo3 import (
    Olmo3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.olmoe import (
    OlmoeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.openelm import (
    OpenElmArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.xglm import (
    XGLMArchitectureAdapter,
)

__all__ = [
    "ApertusArchitectureAdapter",
    "BertArchitectureAdapter",
    "BloomArchitectureAdapter",
    "CodeGenArchitectureAdapter",
    "FalconArchitectureAdapter",
    "Gemma1ArchitectureAdapter",
    "Gemma2ArchitectureAdapter",
    "Gemma3ArchitectureAdapter",
    "Gemma3MultimodalArchitectureAdapter",
    "GraniteArchitectureAdapter",
    "GraniteMoeArchitectureAdapter",
    "GraniteMoeHybridArchitectureAdapter",
    "GPT2ArchitectureAdapter",
    "GPTOSSArchitectureAdapter",
    "Gpt2LmHeadCustomArchitectureAdapter",
    "GptjArchitectureAdapter",
    "HubertArchitectureAdapter",
    "LlamaArchitectureAdapter",
    "LlavaArchitectureAdapter",
    "LlavaNextArchitectureAdapter",
    "LlavaOnevisionArchitectureAdapter",
    "MingptArchitectureAdapter",
    "MistralArchitectureAdapter",
    "MixtralArchitectureAdapter",
    "NanogptArchitectureAdapter",
    "NeelSoluOldArchitectureAdapter",
    "NeoArchitectureAdapter",
    "NeoxArchitectureAdapter",
    "OpenElmArchitectureAdapter",
    "OlmoArchitectureAdapter",
    "Olmo2ArchitectureAdapter",
    "Olmo3ArchitectureAdapter",
    "OlmoeArchitectureAdapter",
    "OptArchitectureAdapter",
    "PhiArchitectureAdapter",
    "Phi3ArchitectureAdapter",
    "PythiaArchitectureAdapter",
    "QwenArchitectureAdapter",
    "Qwen2ArchitectureAdapter",
    "Qwen3ArchitectureAdapter",
    "StableLmArchitectureAdapter",
    "T5ArchitectureAdapter",
    "XGLMArchitectureAdapter",
]
