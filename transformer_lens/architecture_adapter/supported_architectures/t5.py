"""T5 architecture adapter."""

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.conversion_utils.conversion_steps import (
    RearrangeWeightConversion,
    WeightConversionSet,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class T5ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5 models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the T5 architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "shared.weight",
                "blocks.{i}.ln1.w": "encoder.block.{i}.layer.0.layer_norm.weight",
                "blocks.{i}.ln1.b": "encoder.block.{i}.layer.0.layer_norm.bias",
                "blocks.{i}.ln2.w": "encoder.block.{i}.layer.1.layer_norm.weight",
                "blocks.{i}.ln2.b": "encoder.block.{i}.layer.1.layer_norm.bias",
                "blocks.{i}.attn.W_Q": (
                    "encoder.block.{i}.layer.0.SelfAttention.q.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "encoder.block.{i}.layer.0.SelfAttention.k.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "encoder.block.{i}.layer.0.SelfAttention.v.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "encoder.block.{i}.layer.0.SelfAttention.q.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "encoder.block.{i}.layer.0.SelfAttention.k.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "encoder.block.{i}.layer.0.SelfAttention.v.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "encoder.block.{i}.layer.0.SelfAttention.o.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "encoder.block.{i}.layer.0.SelfAttention.o.bias",
                "blocks.{i}.mlp.W_in": "encoder.block.{i}.layer.1.DenseReluDense.wi.weight",
                "blocks.{i}.mlp.b_in": "encoder.block.{i}.layer.1.DenseReluDense.wi.bias",
                "blocks.{i}.mlp.W_out": "encoder.block.{i}.layer.1.DenseReluDense.wo.weight",
                "blocks.{i}.mlp.b_out": "encoder.block.{i}.layer.1.DenseReluDense.wo.bias",
                "unembed.W_U": "shared.weight",
                "unembed.b_U": "shared.bias",
                "ln_final.w": "encoder.final_layer_norm.weight",
                "ln_final.b": "encoder.final_layer_norm.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": "shared",  # Word token embeddings (shared with unembed)
            "blocks": (
                "encoder.block",  # Base path for blocks
                {
                    "ln1": "layer.0.layer_norm",  # Pre-attention layer norm
                    "ln2": "layer.1.layer_norm",  # Pre-MLP layer norm
                    "attn": "layer.0.SelfAttention",  # Full attention module
                    "attn.q_proj": "layer.0.SelfAttention.q",  # Query projection
                    "attn.k_proj": "layer.0.SelfAttention.k",  # Key projection
                    "attn.v_proj": "layer.0.SelfAttention.v",  # Value projection
                    "attn.output_proj": "layer.0.SelfAttention.o",  # Output projection
                    "mlp": "layer.1.DenseReluDense",  # Full MLP module
                    "mlp.fc1": "layer.1.DenseReluDense.wi",  # First linear layer
                    "mlp.fc2": "layer.1.DenseReluDense.wo",  # Second linear layer
                },
            ),
            "ln_final": "encoder.final_layer_norm",  # Final layer norm
            "unembed": "shared",  # Language model head (shared with embed)
        }
