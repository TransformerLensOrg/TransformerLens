"""NeoX architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    SplitWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.conversion_utils.conversion_steps.chain_weight_conversion import (
    ChainWeightConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


class NeoxArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NeoX models."""

    def __init__(self, user_cfg: Any) -> None:
        """Initialize the NeoX architecture adapter.

        Args:
            user_cfg: The configuration object.
        """
        super().__init__(user_cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "gpt_neox.embed_in.weight",
                "blocks.{i}.ln1.w": "gpt_neox.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "gpt_neox.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "gpt_neox.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "gpt_neox.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainWeightConversion(
                        [
                            SplitWeightConversion(0, 3),
                            RearrangeWeightConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.user_cfg.num_attention_heads,
                                d_head=self.user_cfg.hidden_size
                                // self.user_cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainWeightConversion(
                        [
                            SplitWeightConversion(1, 3),
                            RearrangeWeightConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.user_cfg.num_attention_heads,
                                d_head=self.user_cfg.hidden_size
                                // self.user_cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    ChainWeightConversion(
                        [
                            SplitWeightConversion(2, 3),
                            RearrangeWeightConversion(
                                "(head d_head) d_model -> head d_model d_head",
                                head=self.user_cfg.num_attention_heads,
                                d_head=self.user_cfg.hidden_size
                                // self.user_cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainWeightConversion(
                        [
                            SplitWeightConversion(0, 3),
                            RearrangeWeightConversion(
                                "(head d_head) -> head d_head",
                                head=self.user_cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_K": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainWeightConversion(
                        [
                            SplitWeightConversion(1, 3),
                            RearrangeWeightConversion(
                                "(head d_head) -> head d_head",
                                head=self.user_cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.b_V": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    ChainWeightConversion(
                        [
                            SplitWeightConversion(2, 3),
                            RearrangeWeightConversion(
                                "(head d_head) -> head d_head",
                                head=self.user_cfg.num_attention_heads,
                            ),
                        ]
                    ),
                ),
                "blocks.{i}.attn.W_O": (
                    "gpt_neox.layers.{i}.attention.dense.weight",
                    RearrangeWeightConversion("d_model (head d_head) -> head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "gpt_neox.layers.{i}.attention.dense.bias",
                "blocks.{i}.mlp.W_in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                "blocks.{i}.mlp.b_in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                "blocks.{i}.mlp.W_out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                "blocks.{i}.mlp.b_out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                "ln_final.w": "gpt_neox.final_layer_norm.weight",
                "ln_final.b": "gpt_neox.final_layer_norm.bias",
                "unembed.W_U": "embed_out.weight",
            }
        )

        self.component_mapping = {
            "embed": ("gpt_neox.embed_in", EmbeddingBridge),
            "pos_embed": ("gpt_neox.embed_pos", EmbeddingBridge),
            "blocks": (
                "gpt_neox.layers",
                BlockBridge,
                {
                    "ln1": ("input_layernorm", LayerNormBridge),
                    "ln2": ("post_attention_layernorm", LayerNormBridge),
                    "attn": ("attention", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            ),
            "ln_final": ("gpt_neox.final_layer_norm", LayerNormBridge),
            "unembed": ("embed_out", UnembeddingBridge),
        }
