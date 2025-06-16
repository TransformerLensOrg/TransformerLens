"""Qwen2 architecture adapter."""

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


class Qwen2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen2 models."""

    def __init__(self, user_cfg: Any) -> None:
        """Initialize the Qwen2 architecture adapter."""
        super().__init__(user_cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion(
                        "(n h) m -> n m h", n=self.user_cfg.num_attention_heads
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion(
                        "(n h) m -> n m h",
                        n=getattr(
                            self.user_cfg, "num_key_value_heads", self.user_cfg.num_attention_heads
                        ),
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion(
                        "(n h) m -> n m h",
                        n=getattr(
                            self.user_cfg, "num_key_value_heads", self.user_cfg.num_attention_heads
                        ),
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion(
                        "m (n h) -> n h m", n=self.user_cfg.num_attention_heads
                    ),
                ),
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight",
            }
        )
        self.component_mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),
            "blocks": (
                "model.layers",
                BlockBridge,
                {
                    "ln1": ("input_layernorm", LayerNormBridge),
                    "ln2": ("post_attention_layernorm", LayerNormBridge),
                    "attn": ("self_attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            ),
            "ln_final": ("model.norm", LayerNormBridge),
            "unembed": ("lm_head", UnembeddingBridge),
        }
