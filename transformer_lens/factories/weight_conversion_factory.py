from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.bert import BertWeightConversion
from transformer_lens.weight_conversion.bloom import BloomWeightConversion
from transformer_lens.weight_conversion.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.weight_conversion.gemma import GemmaWeightConversion
from transformer_lens.weight_conversion.gpt2 import GPT2WeightConversion
from transformer_lens.weight_conversion.gpt2_lm_head_custom import (
    GPT2LMHeadCustomWeightConversion,
)
from transformer_lens.weight_conversion.gptj import GPTJWeightConversion
from transformer_lens.weight_conversion.mistral import MistralWeightConversion
from transformer_lens.weight_conversion.mixtral import MixtralWeightConversion
from transformer_lens.weight_conversion.neo import NEOWeightConversion
from transformer_lens.weight_conversion.neox import NEOXWeightConversion
from transformer_lens.weight_conversion.qwen import QwenWeightConversion
from transformer_lens.weight_conversion.qwen2 import Qwen2WeightConversion
from transformer_lens.weight_conversion.t5 import T5WeightConversion
from transformer_lens.weight_conversion.phi import PhiWeightConversion

class WeightConversionFactory:
    @staticmethod
    def select_weight_conversion_config(cfg: HookedTransformerConfig) -> ArchitectureConversion:
        match cfg.original_architecture:
            case "BertForMaskedLM":
                return BertWeightConversion(cfg)
            case "BloomForCausalLM":
                return BloomWeightConversion(cfg)
            case "Gemma2ForCausalLM":
                return GemmaWeightConversion(cfg)
            case "GPT2LMHeadModel":
                return GPT2WeightConversion(cfg)
            case "GPT2LMHeadCustomModel":
                return GPT2LMHeadCustomWeightConversion(cfg)
            case "GPTJForCausalLM":
                return GPTJWeightConversion(cfg)
            case "GPTNeoForCausalLM":
                return NEOWeightConversion(cfg)
            case "GPTNeoXForCausalLM":
                return NEOXWeightConversion(cfg)
            case "MistralForCausalLM":
                return MistralWeightConversion(cfg)
            case "MixtralForCausalLM":
                return MixtralWeightConversion(cfg)
            case "PhiForCausalLM":
                return PhiWeightConversion(cfg)
            case "Qwen2ForCausalLM":
                return Qwen2WeightConversion(cfg)
            case "QWenLMHeadModel":
                return QwenWeightConversion(cfg)
            case "T5ForConditionalGeneration":
                return T5WeightConversion(cfg)
            case _:
                raise NotImplementedError(
                    f"{cfg.original_architecture} is not currently supported."
                )
