"""Supported architecture adapters.

This module contains all the supported architecture adapters for different model architectures.
"""

from transformer_lens.model_bridge.supported_architectures.apertus import (
    ApertusArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.arcee import (
    ArceeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.bamba import (
    BambaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.baichuan import (
    BaichuanArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.bart import (
    BartArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.bd3lm import (
    BD3LMArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.cohere import (
    Cohere2ArchitectureAdapter,
    CohereArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekV2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v3 import (
    DeepSeekV3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.exaone import (
    ExaoneArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.falcon import (
    FalconArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.falcon_mamba import (
    FalconMambaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.falcon_h1 import (
    FalconH1ArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.gemma3n import (
    Gemma3nArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gemma4 import (
    Gemma4ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.glm4_moe import (
    Glm4MoeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.glm4_moe_lite import (
    Glm4MoeLiteArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.glm_moe_dsa import (
    GlmMoeDsaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt2_lm_head_custom import (
    Gpt2LmHeadCustomArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gpt_bigcode import (
    GPTBigCodeArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.hunyuan_v1_dense import (
    HunYuanDenseV1ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.internlm2 import (
    InternLM2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.lfm2_moe import (
    Lfm2MoeArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.m2m100 import (
    M2M100ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mamba import (
    MambaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mamba2 import (
    Mamba2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.marian import (
    MarianArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mbart import (
    MBartArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mingpt import (
    MingptArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.minimax_m2 import (
    MiniMaxM2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mixtral import (
    MixtralArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mpt import (
    MPTArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.nanogpt import (
    NanogptArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.native import (
    NativeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.neel_solu_old import (
    NeelSoluOldArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.nemotron_h import (
    NemotronHArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.openai_gpt import (
    OpenAIGPTArchitectureAdapter,
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
from transformer_lens.model_bridge.supported_architectures.phimoe import (
    PhiMoEArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen import (
    QwenArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen2_moe import (
    Qwen2MoeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
    Qwen3_5ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal import (
    Qwen3_5MultimodalArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5_moe import (
    Qwen3_5MoeArchitectureAdapter,
    Qwen3_5MoeMultimodalArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_moe import (
    Qwen3MoeArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_next import (
    Qwen3NextArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.recurrent_gemma import (
    RecurrentGemmaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.smollm3 import (
    SmolLM3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.stablelm import (
    StableLmArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.t5gemma import (
    T5GemmaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.xglm import (
    XGLMArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.zamba2 import (
    Zamba2ArchitectureAdapter,
)

__all__ = [
    "ApertusArchitectureAdapter",
    "ArceeArchitectureAdapter",
    "BD3LMArchitectureAdapter",
    "BaichuanArchitectureAdapter",
    "BambaArchitectureAdapter",
    "BartArchitectureAdapter",
    "BertArchitectureAdapter",
    "BloomArchitectureAdapter",
    "CodeGenArchitectureAdapter",
    "Cohere2ArchitectureAdapter",
    "CohereArchitectureAdapter",
    "DeepSeekV2ArchitectureAdapter",
    "DeepSeekV3ArchitectureAdapter",
    "ExaoneArchitectureAdapter",
    "FalconArchitectureAdapter",
    "FalconH1ArchitectureAdapter",
    "FalconMambaArchitectureAdapter",
    "Gemma1ArchitectureAdapter",
    "Gemma2ArchitectureAdapter",
    "Gemma3ArchitectureAdapter",
    "Gemma3nArchitectureAdapter",
    "Gemma3MultimodalArchitectureAdapter",
    "Gemma4ArchitectureAdapter",
    "GlmMoeDsaArchitectureAdapter",
    "Glm4MoeArchitectureAdapter",
    "Glm4MoeLiteArchitectureAdapter",
    "GraniteArchitectureAdapter",
    "GraniteMoeArchitectureAdapter",
    "GraniteMoeHybridArchitectureAdapter",
    "GPT2ArchitectureAdapter",
    "GPTBigCodeArchitectureAdapter",
    "GPTOSSArchitectureAdapter",
    "Gpt2LmHeadCustomArchitectureAdapter",
    "GptjArchitectureAdapter",
    "HubertArchitectureAdapter",
    "HunYuanDenseV1ArchitectureAdapter",
    "InternLM2ArchitectureAdapter",
    "LlamaArchitectureAdapter",
    "LlavaArchitectureAdapter",
    "LlavaNextArchitectureAdapter",
    "LlavaOnevisionArchitectureAdapter",
    "Lfm2MoeArchitectureAdapter",
    "M2M100ArchitectureAdapter",
    "MambaArchitectureAdapter",
    "Mamba2ArchitectureAdapter",
    "MarianArchitectureAdapter",
    "MBartArchitectureAdapter",
    "MingptArchitectureAdapter",
    "MiniMaxM2ArchitectureAdapter",
    "MistralArchitectureAdapter",
    "MixtralArchitectureAdapter",
    "MPTArchitectureAdapter",
    "NanogptArchitectureAdapter",
    "NemotronHArchitectureAdapter",
    "NativeArchitectureAdapter",
    "NeelSoluOldArchitectureAdapter",
    "NemotronHArchitectureAdapter",
    "NeoArchitectureAdapter",
    "NeoxArchitectureAdapter",
    "OpenAIGPTArchitectureAdapter",
    "OpenElmArchitectureAdapter",
    "OlmoArchitectureAdapter",
    "Olmo2ArchitectureAdapter",
    "Olmo3ArchitectureAdapter",
    "OlmoeArchitectureAdapter",
    "OptArchitectureAdapter",
    "PhiArchitectureAdapter",
    "Phi3ArchitectureAdapter",
    "PhiMoEArchitectureAdapter",
    "QwenArchitectureAdapter",
    "Qwen2ArchitectureAdapter",
    "Qwen2MoeArchitectureAdapter",
    "Qwen3ArchitectureAdapter",
    "Qwen3MoeArchitectureAdapter",
    "Qwen3NextArchitectureAdapter",
    "Qwen3_5ArchitectureAdapter",
    "Qwen3_5MoeArchitectureAdapter",
    "Qwen3_5MoeMultimodalArchitectureAdapter",
    "Qwen3_5MultimodalArchitectureAdapter",
    "RecurrentGemmaArchitectureAdapter",
    "SmolLM3ArchitectureAdapter",
    "StableLmArchitectureAdapter",
    "T5ArchitectureAdapter",
    "T5GemmaArchitectureAdapter",
    "XGLMArchitectureAdapter",
    "Zamba2ArchitectureAdapter",
]
