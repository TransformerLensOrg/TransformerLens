"""Phi architecture adapter."""

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class PhiArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Phi models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Phi architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.embd.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.q_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.k_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.v_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.q_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.k_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.v_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.out_proj.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.out_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.fc1.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.fc1.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.fc2.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.fc2.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": "transformer.embd.wte",  # Word token embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": "ln_1",  # Pre-attention layer norm
                    "ln2": "ln_2",  # Pre-MLP layer norm
                    "attn": "attn",  # Full attention module
                    "attn.q_proj": "attn.q_proj",  # Query projection
                    "attn.k_proj": "attn.k_proj",  # Key projection
                    "attn.v_proj": "attn.v_proj",  # Value projection
                    "attn.output_proj": "attn.out_proj",  # Output projection
                    "mlp": "mlp",  # Full MLP module
                    "mlp.fc1": "mlp.fc1",  # First linear layer
                    "mlp.fc2": "mlp.fc2",  # Second linear layer
                },
            ),
            "ln_final": "transformer.ln_f",  # Final layer norm
            "unembed": "lm_head",  # Language model head
        }
