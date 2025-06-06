"""T5 architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
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


class T5ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5 models."""

    def __init__(self, user_cfg: Any) -> None:
        """Initialize the T5 architecture adapter.

        Args:
            user_cfg: The configuration object.
        """
        super().__init__(user_cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "shared.weight",
                "pos_embed.W_pos": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "blocks.{i}.ln1.w": "encoder.block.{i}.layer.0.layer_norm.weight",
                "blocks.{i}.attn.W_Q": "encoder.block.{i}.layer.0.SelfAttention.q.weight",
                "blocks.{i}.attn.W_K": "encoder.block.{i}.layer.0.SelfAttention.k.weight",
                "blocks.{i}.attn.W_V": "encoder.block.{i}.layer.0.SelfAttention.v.weight",
                "blocks.{i}.attn.W_O": "encoder.block.{i}.layer.0.SelfAttention.o.weight",
                "blocks.{i}.ln2.w": "encoder.block.{i}.layer.1.layer_norm.weight",
                "blocks.{i}.mlp.W_in": "encoder.block.{i}.layer.1.DenseReluDense.wi.weight",
                "blocks.{i}.mlp.W_out": "encoder.block.{i}.layer.1.DenseReluDense.wo.weight",
                "ln_final.w": "encoder.final_layer_norm.weight",
                "unembed.W_U": "lm_head.weight",
            }
        )
        self.component_mapping = {
            "embed": ("shared", EmbeddingBridge),
            "pos_embed": (
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias",
                EmbeddingBridge,
            ),
            "blocks": (
                "encoder.block",
                BlockBridge,
                {
                    "ln1": ("layer.0.layer_norm", LayerNormBridge),
                    "attn": ("layer.0.SelfAttention", AttentionBridge),
                    "ln2": ("layer.1.layer_norm", LayerNormBridge),
                    "mlp": ("layer.1.DenseReluDense", MLPBridge),
                },
            ),
            "ln_final": ("encoder.final_layer_norm", LayerNormBridge),
            "unembed": ("lm_head", UnembeddingBridge),
        }
