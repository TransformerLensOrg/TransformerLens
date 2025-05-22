"""Gemma2 architecture adapter."""

from transformer_lens.architecture_adapter.architecture_adapter import (
    ArchitectureAdapter,
)
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
from transformer_lens.model_bridge.model_bridge import ModelBridge
from transformer_lens.TransformerLensConfig import TransformerLensConfig


class Gemma2ArchitectureAdapter(ModelBridge):
    """Architecture adapter for Gemma2 models."""

    def __init__(self, cfg: TransformerLensConfig) -> None:
        """Initialize the Gemma2 architecture adapter.

        Args:
            cfg: The TransformerLens configuration.
        """
        super().__init__(cfg)

        # Set up weight conversion rules
        self.conversion_rules = WeightConversionSet(
            {
                # Gemma2 scales embeddings by sqrt(d_model)
                "embed.W_E": (
                    "model.embed_tokens.weight",
                    RearrangeWeightConversion(
                        "d_vocab d_model -> d_vocab d_model",
                        scale=cfg.d_model**0.5,
                    ),
                ),
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1_post.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_heads),
                ),
                "blocks.{i}.attn._W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads),
                ),
                "blocks.{i}.attn._W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(n h) m->n m h", n=cfg.n_key_value_heads),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("m (n h)->n h m", n=cfg.n_heads),
                ),
                "blocks.{i}.ln2.w": "model.layers.{i}.pre_feedforward_layernorm.weight",
                "blocks.{i}.ln2_post.w": "model.layers.{i}.post_feedforward_layernorm.weight",
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.W_gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.W_U": "lm_head.weight.T",  # Not shared with embedding
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "model.layers",  # Base path for blocks
                {
                    "ln1": ("input_layernorm", LayerNormBridge),  # Pre-attention layer norm
                    "ln1_post": ("post_attention_layernorm", LayerNormBridge),  # Post-attention layer norm
                    "attn": ("self_attn", AttentionBridge),  # Full attention module
                    "ln2": ("pre_feedforward_layernorm", LayerNormBridge),  # Pre-MLP layer norm
                    "ln2_post": ("post_feedforward_layernorm", LayerNormBridge),  # Post-MLP layer norm
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                },
            ),
            "ln_final": ("model.norm", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head (not shared with embed)
        } 