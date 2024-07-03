from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.components.mlps.gated_mlp_4bit import GatedMLP4Bit
from transformer_lens.components.mlps.mlp import MLP
from transformer_lens.components.mlps.moe import MoE

class MLPFactory:
    
    @staticmethod
    def create_mlp(config: HookedTransformerConfig) -> CanBeUsedAsMLP:

        if config.num_experts:
            return MoE(config)
        elif config.gated_mlp:
            return GatedMLP(config) if not config.load_in_4bit else GatedMLP4Bit(config)
        else:
            return MLP(config)