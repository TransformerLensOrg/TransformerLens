"""Llama architecture adapter."""

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
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


class LlamaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Llama models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Llama architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        # Set up weight conversion rules
        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln1.b": "model.layers.{i}.input_layernorm.bias",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.ln2.b": "model.layers.{i}.post_attention_layernorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.mlp.W_in": "model.layers.{i}.mlp.gate_proj.weight",
                "blocks.{i}.mlp.b_in": "model.layers.{i}.mlp.gate_proj.bias",
                "blocks.{i}.mlp.W_out": "model.layers.{i}.mlp.down_proj.weight",
                "blocks.{i}.mlp.b_out": "model.layers.{i}.mlp.down_proj.bias",
                "ln_final.w": "model.norm.weight",
                "ln_final.b": "model.norm.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "model.layers",  # Base path for blocks
                {
                    "ln1": ("input_layernorm", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("post_attention_layernorm", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("self_attn", AttentionBridge),  # Full attention module
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                },
            ),
            "ln_final": ("model.norm", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head
        }
