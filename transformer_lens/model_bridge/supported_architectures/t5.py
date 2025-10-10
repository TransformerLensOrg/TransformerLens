"""T5 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import HookConversionSet
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class T5ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the T5 architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "shared.weight",
                "pos_embed.pos": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "blocks.{i}.ln1.w": "encoder.block.{i}.layer.0.layer_norm.weight",
                "blocks.{i}.attn.q": "encoder.block.{i}.layer.0.SelfAttention.q.weight",
                "blocks.{i}.attn.k": "encoder.block.{i}.layer.0.SelfAttention.k.weight",
                "blocks.{i}.attn.v": "encoder.block.{i}.layer.0.SelfAttention.v.weight",
                "blocks.{i}.attn.o": "encoder.block.{i}.layer.0.SelfAttention.o.weight",
                "blocks.{i}.ln2.w": "encoder.block.{i}.layer.1.layer_norm.weight",
                "blocks.{i}.mlp.in": "encoder.block.{i}.layer.1.DenseReluDense.wi.weight",
                "blocks.{i}.mlp.out": "encoder.block.{i}.layer.1.DenseReluDense.wo.weight",
                "ln_final.w": "encoder.final_layer_norm.weight",
                "unembed.u": "lm_head.weight",
            }
        )
        self.component_mapping = {
            "embed": EmbeddingBridge(name="shared"),
            "pos_embed": PosEmbedBridge(
                name="encoder.block.0.layer.0.SelfAttention.relative_attention_bias"
            ),
            "blocks": BlockBridge(
                name="encoder.block",
                submodules={
                    "ln1": NormalizationBridge(name="layer.0.layer_norm", config=self.cfg),
                    "attn": AttentionBridge(name="layer.0.SelfAttention", config=self.cfg),
                    "ln2": NormalizationBridge(name="layer.1.layer_norm", config=self.cfg),
                    "mlp": MLPBridge(name="layer.1.DenseReluDense"),
                },
            ),
            "ln_final": NormalizationBridge(name="encoder.final_layer_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
