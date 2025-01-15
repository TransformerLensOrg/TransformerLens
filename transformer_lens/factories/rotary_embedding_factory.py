from transformer_lens.components.rotary_embeddings import RotaryEmbedding, DynamicNTKScalingRotary
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class RotaryEmbeddingFactory:
    @staticmethod
    def create_rotary(cfg: HookedTransformerConfig) -> RotaryEmbedding:
        if cfg.use_NTK_by_parts_rope:
            return DynamicNTKScalingRotary(cfg)
        else:
            return RotaryEmbedding(cfg)
