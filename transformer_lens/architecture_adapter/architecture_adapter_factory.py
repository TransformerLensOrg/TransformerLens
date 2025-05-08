from transformer_lens.architecture_adapter.bert import BertArchitectureAdapter
from transformer_lens.architecture_adapter.bloom import BloomArchitectureAdapter
from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
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
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class ArchitectureAdapterFactory:
    @staticmethod
    def select_architecture_adapter(cfg: HookedTransformerConfig) -> ArchitectureConversion:
        match cfg.original_architecture:
            case "BertForMaskedLM":
                return BertArchitectureAdapter(cfg)
            case "BloomForCausalLM":
                return BloomArchitectureAdapter(cfg)
            case "Gemma2ForCausalLM":
                return GemmaArchitectureAdapter(cfg)
            case "GPT2LMHeadModel":
                return GPT2ArchitectureAdapter(cfg)
            case "GPT2LMHeadCustomModel":
                return GPT2LMHeadCustomArchitectureAdapter(cfg)
            case "GPTJForCausalLM":
                return GPTJArchitectureAdapter(cfg)
            case "GPTNeoForCausalLM":
                return NEOArchitectureAdapter(cfg)
            case "GPTNeoXForCausalLM":
                return NEOXArchitectureAdapter(cfg)
            case "LlamaForCausalLM":
                return LLAMAArchitectureAdapter(cfg)
            case "MistralForCausalLM":
                return MistralArchitectureAdapter(cfg)
            case "MixtralForCausalLM":
                return MixtralArchitectureAdapter(cfg)
            case "PhiForCausalLM":
                return PhiArchitectureAdapter(cfg)
            case "Qwen2ForCausalLM":
                return Qwen2ArchitectureAdapter(cfg)
            case "QWenLMHeadModel":
                return QwenArchitectureAdapter(cfg)
            case "T5ForConditionalGeneration":
                return T5ArchitectureAdapter(cfg)
            case _:
                raise NotImplementedError(
                    f"{cfg.original_architecture} is not currently supported."
                ) 