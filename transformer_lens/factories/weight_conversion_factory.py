from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.weight_conversion.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)
from transformer_lens.weight_conversion.mixtral import MixtralWeightConversion


class WeightConversionFactory:
    @staticmethod
    def select_weight_conversion_config(cfg: HookedTransformerConfig) -> ArchitectureConversion:
        match cfg.original_architecture:
            case "MixtralForCausalLM":
                return MixtralWeightConversion(cfg)
            case _:
                raise NotImplementedError(
                    f"{cfg.original_architecture} is not currently supported."
                )
