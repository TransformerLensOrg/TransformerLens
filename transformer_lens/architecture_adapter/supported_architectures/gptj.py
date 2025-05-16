"""GPT-J architecture adapter."""

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


class GPTJArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-J models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the GPT-J architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.q_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.k_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.v_proj.weight",
                    RearrangeWeightConversion("d_model (n_head d_head) -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.out_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.fc_in.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.fc_in.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.fc_out.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.fc_out.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("transformer.wte", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": ("ln_1", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("ln_2", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("attn", AttentionBridge),  # Full attention module
                    "attn.q_proj": ("attn.q_proj", AttentionBridge),  # Query projection
                    "attn.k_proj": ("attn.k_proj", AttentionBridge),  # Key projection
                    "attn.v_proj": ("attn.v_proj", AttentionBridge),  # Value projection
                    "attn.out_proj": ("attn.out_proj", AttentionBridge),  # Output projection
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                    "mlp.fc_in": ("mlp.fc_in", MLPBridge),  # First linear layer
                    "mlp.fc_out": ("mlp.fc_out", MLPBridge),  # Second linear layer
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head
        }
