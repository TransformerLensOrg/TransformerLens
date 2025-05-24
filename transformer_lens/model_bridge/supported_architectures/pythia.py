"""Pythia architecture adapter."""

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)


class PythiaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Pythia models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Pythia architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "gpt_neox.embed_in.weight",
                "pos_embed.W_pos": "gpt_neox.embed_pos.weight",
                "blocks.{i}.ln1.w": "gpt_neox.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "gpt_neox.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "gpt_neox.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "gpt_neox.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    RearrangeWeightConversion(
                        "(3 n_head d_head) d_model -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    RearrangeWeightConversion(
                        "(3 n_head d_head) d_model -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "gpt_neox.layers.{i}.attention.query_key_value.weight",
                    RearrangeWeightConversion(
                        "(3 n_head d_head) d_model -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "gpt_neox.layers.{i}.attention.query_key_value.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "gpt_neox.layers.{i}.attention.dense.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "gpt_neox.layers.{i}.attention.dense.bias",
                "blocks.{i}.mlp.W_in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",
                "blocks.{i}.mlp.b_in": "gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias",
                "blocks.{i}.mlp.W_out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",
                "blocks.{i}.mlp.b_out": "gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias",
                "ln_final.w": "gpt_neox.final_layer_norm.weight",
                "ln_final.b": "gpt_neox.final_layer_norm.bias",
                "unembed.W_U": "embed_out.weight",
                "unembed.b_U": "embed_out.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("gpt_neox.embed_in", EmbeddingBridge),  # Word token embeddings
            "pos_embed": ("gpt_neox.embed_pos", EmbeddingBridge),  # Position embeddings
            "blocks": (
                "gpt_neox.layers",  # Base path for blocks
                {
                    "ln1": ("ln_1", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("ln_2", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("attn", AttentionBridge),  # Full attention module
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),  # Final layer norm
            "unembed": ("embed_out", UnembeddingBridge),  # Language model head
        }
