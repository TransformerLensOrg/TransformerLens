"""BLOOM architecture adapter."""

from typing import Any, Dict

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class BloomArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BLOOM models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the BLOOM architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.word_embeddings.weight",
                "pos_embed.W_pos": "transformer.word_embeddings_layernorm.weight",
                "embed.LayerNorm.weight": "transformer.word_embeddings_layernorm.weight",
                "embed.LayerNorm.bias": "transformer.word_embeddings_layernorm.bias",
                "blocks.{i}.ln1.w": "transformer.h.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.self_attention.query_key_value.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.self_attention.query_key_value.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.self_attention.query_key_value.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.self_attention.query_key_value.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.self_attention.dense.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.self_attention.dense.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.dense_h_to_4h.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.dense_h_to_4h.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.dense_4h_to_h.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.dense_4h_to_h.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
            }
        )
