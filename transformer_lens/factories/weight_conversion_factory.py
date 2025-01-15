from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.weight_conversion.bert import BertWeightConversion
from transformer_lens.weight_conversion.gemma import GemmaWeightConversion
from transformer_lens.weight_conversion.gpt2_lm_head_custom import GPT2LMHeadCustomWeightConversion
from transformer_lens.weight_conversion.mixtral import MixtralWeightConversion
from transformer_lens.weight_conversion.qwen import QwenWeightConversion
from transformer_lens.weight_conversion.qwen2 import Qwen2WeightConversion


class WeightConversionFactory:
    @staticmethod
    def select_weight_conversion_config(cfg: HookedTransformerConfig) -> ArchitectureConversion:
        match cfg.original_architecture:
            case "BertForMaskedLM":
                return BertWeightConversion(cfg)
            case "MixtralForCausalLM":
                return MixtralWeightConversion(cfg)
            case "Gemma2ForCausalLM":
                return GemmaWeightConversion(cfg)
            case "Qwen2ForCausalLM":
                return Qwen2WeightConversion(cfg)
            case "QWenLMHeadModel":
                return QwenWeightConversion(cfg)
            case "GPT2LMHeadCustomModel":
                return GPT2LMHeadCustomWeightConversion(cfg)
            case _:
                raise NotImplementedError(
                    f"{cfg.original_architecture} is not currently supported."
                )
