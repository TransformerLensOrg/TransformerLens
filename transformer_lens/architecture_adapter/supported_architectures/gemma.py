"""Gemma architecture adapter."""

from typing import Any, Dict

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class GemmaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Gemma architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "model.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "model.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "model.layers.{i}.self_attn.q_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "model.layers.{i}.self_attn.k_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "model.layers.{i}.self_attn.v_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "model.layers.{i}.self_attn.o_proj.bias",
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.b_in": "model.layers.{i}.mlp.gate_proj.bias",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight",
                "blocks.{i}.mlp.b_out": "model.layers.{i}.mlp.down_proj.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "model.norm.weight",
                "ln_final.b": "model.norm.bias",
            }
        )
