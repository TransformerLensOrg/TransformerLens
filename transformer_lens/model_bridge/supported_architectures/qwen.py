"""Qwen architecture adapter."""

from typing import Any, Dict

from transformer_lens.architecture_adapter.architecture_adapter import ArchitectureAdapter
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.architecture_adapter.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class QwenArchitectureAdapter(ModelBridge):
    """Architecture adapter for Qwen models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Qwen architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        self.default_config = {"trust_remote_code": True}
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "pos_embed.W_pos": "transformer.wpe.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.q_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.k_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.v_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.q_proj.bias",
                    RearrangeWeightConversion("(n_head d_head) -> n_head d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.k_proj.bias",
                    RearrangeWeightConversion("(n_head d_head) -> n_head d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.v_proj.bias",
                    RearrangeWeightConversion("(n_head d_head) -> n_head d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.o_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.o_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.gate_proj.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.down_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.down_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("transformer.wte", EmbeddingBridge),  # Word token embeddings
            "pos_embed": ("transformer.wpe", EmbeddingBridge),  # Position embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": ("ln_1", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("ln_2", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("attn", AttentionBridge),  # Full attention module
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head
        }
