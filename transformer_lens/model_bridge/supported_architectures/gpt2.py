"""GPT-2 architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-2 models."""

    def __init__(self, user_cfg: Any) -> None:
        """Initialize the GPT-2 architecture adapter."""
        super().__init__(user_cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "pos_embed.W_pos": "transformer.wpe.weight",
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("d_model (n d_head) -> n d_model d_head"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("d_model (n d_head) -> n d_model d_head"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("d_model (n d_head) -> n d_model d_head"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(n d_head) -> n d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(n d_head) -> n d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(n d_head) -> n d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("(n d_head) d_model -> n d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        self.component_mapping = {
            "embed": ("transformer.wte", EmbeddingBridge),
            "pos_embed": ("transformer.wpe", EmbeddingBridge),
            "blocks": (
                "transformer.h",
                BlockBridge,
                {
                    "ln1": ("ln_1", LayerNormBridge),
                    "attn": ("attn", AttentionBridge),
                    "ln2": ("ln_2", LayerNormBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),
            "unembed": ("lm_head", UnembeddingBridge),
        }
