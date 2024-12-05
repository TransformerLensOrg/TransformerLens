from transformer_lens.weight_conversion.conversion_utils.architecture_conversion import ArchitectureConversion
from transformer_lens.weight_conversion.mixtral import MixtralWeightConversion
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

class MLPFactory:
    @staticmethod
    def select_weight_conversion_config(cfg: HookedTransformerConfig, architecture: str) -> ArchitectureConversion:
        match architecture:
            case "MixtralForCausalLM":
                return MixtralWeightConversion(cfg)
            case _:
                raise NotImplementedError(f"{architecture} is not currently supported.")