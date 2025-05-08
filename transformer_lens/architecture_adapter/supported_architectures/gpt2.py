"""GPT-2 architecture adapter."""

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class GPT2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-2 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the GPT-2 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        # Set up weight conversion rules
        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(3 h d_head) d_model -> 3 h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(3 h d_head) d_model -> 3 h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion("(3 h d_head) d_model -> 3 h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 h d_head) -> 3 h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 h d_head) -> 3 h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 h d_head) -> 3 h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": "transformer.wte",  # Word token embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": "ln_1",  # Pre-attention layer norm
                    "ln2": "ln_2",  # Pre-MLP layer norm
                    "attn": "attn",  # Full attention module
                    "attn.qkv_proj": "attn.c_attn",  # Combined QKV projection
                    "attn.output_proj": "attn.c_proj",  # Output projection
                    "mlp": "mlp",  # Full MLP module
                    "mlp.fc1": "mlp.c_fc",  # First linear layer
                    "mlp.fc2": "mlp.c_proj",  # Second linear layer
                    "mlp.act_fn": "mlp.act",  # Activation function
                },
            ),
            "ln_final": "transformer.ln_f",  # Final layer norm
            "unembed": "lm_head",  # Language model head
        }
