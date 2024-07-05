"""MLP Factory

Centralized location for creating any MLP needed within TransformerLens
"""
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.components.mlps.gated_mlp_4bit import GatedMLP4Bit
from transformer_lens.components.mlps.mlp import MLP
from transformer_lens.components.mlps.moe import MoE
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class MLPFactory:
    @staticmethod
    def create_mlp(cfg: HookedTransformerConfig) -> CanBeUsedAsMLP:
        if cfg.num_experts:
            return MoE(cfg)
        elif cfg.gated_mlp:
            return GatedMLP(cfg) if not cfg.load_in_4bit else GatedMLP4Bit(cfg)
        else:
            return MLP(cfg)
