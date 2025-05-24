"""Gemma3 architecture adapter."""

from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel

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


class Gemma3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma3 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Gemma3 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        # Set up weight conversion rules
        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.num_attention_heads),
                ),
                "blocks.{i}.attn._W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)),
                ),
                "blocks.{i}.attn._W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("m (n h)->n h m", n=cfg.num_attention_heads),
                ),
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "embed_tokens.weight.T",  # Shared with embedding
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "model.layers",  # Base path for blocks
                {
                    "ln1": ("input_layernorm", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("post_attention_layernorm", LayerNormBridge),  # Post-attention layer norm
                    "attn": ("self_attn", AttentionBridge),  # Full attention module
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                },
            ),
            "ln_final": ("model.norm", LayerNormBridge),  # Final layer norm
            "unembed": ("model.embed_tokens", UnembeddingBridge),  # Language model head (shared with embed)
        }