"""BERT architecture adapter."""

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


class BertArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BERT models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the BERT architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "bert.embeddings.word_embeddings.weight",
                "pos_embed.W_pos": "bert.embeddings.position_embeddings.weight",
                "embed.token_type_embeddings": "bert.embeddings.token_type_embeddings.weight",
                "embed.LayerNorm.weight": "bert.embeddings.LayerNorm.weight",
                "embed.LayerNorm.bias": "bert.embeddings.LayerNorm.bias",
                "blocks.{i}.ln1.w": "bert.encoder.layer.{i}.attention.output.LayerNorm.weight",
                "blocks.{i}.ln1.b": "bert.encoder.layer.{i}.attention.output.LayerNorm.bias",
                "blocks.{i}.ln2.w": "bert.encoder.layer.{i}.output.LayerNorm.weight",
                "blocks.{i}.ln2.b": "bert.encoder.layer.{i}.output.LayerNorm.bias",
                "blocks.{i}.attn.W_Q": (
                    "bert.encoder.layer.{i}.attention.self.query.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "bert.encoder.layer.{i}.attention.self.key.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "bert.encoder.layer.{i}.attention.self.value.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "bert.encoder.layer.{i}.attention.self.query.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "bert.encoder.layer.{i}.attention.self.key.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "bert.encoder.layer.{i}.attention.self.value.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "bert.encoder.layer.{i}.attention.output.dense.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "bert.encoder.layer.{i}.attention.output.dense.bias",
                "blocks.{i}.mlp.W_in": "bert.encoder.layer.{i}.intermediate.dense.weight",
                "blocks.{i}.mlp.b_in": "bert.encoder.layer.{i}.intermediate.dense.bias",
                "blocks.{i}.mlp.W_out": "bert.encoder.layer.{i}.output.dense.weight",
                "blocks.{i}.mlp.b_out": "bert.encoder.layer.{i}.output.dense.bias",
                "unembed.W_U": "cls.predictions.transform.dense.weight",
                "unembed.b_U": "cls.predictions.transform.dense.bias",
                "unembed.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
                "unembed.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
                "unembed.decoder.weight": "cls.predictions.decoder.weight",
                "unembed.decoder.bias": "cls.predictions.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("bert.embeddings", EmbeddingBridge),  # Word token embeddings
            "pos_embed": ("bert.embeddings.position_embeddings", EmbeddingBridge),  # Position embeddings
            "blocks": (
                "bert.encoder.layer",  # Base path for blocks
                {
                    "ln1": ("attention.output.LayerNorm", LayerNormBridge),  # Post-attention layer norm
                    "ln2": ("output.LayerNorm", LayerNormBridge),  # Post-MLP layer norm
                    "attn": ("attention", AttentionBridge),  # Full attention module
                    "attn.q_proj": ("attention.self.query", AttentionBridge),  # Query projection
                    "attn.k_proj": ("attention.self.key", AttentionBridge),  # Key projection
                    "attn.v_proj": ("attention.self.value", AttentionBridge),  # Value projection
                    "attn.output_proj": ("attention.output.dense", AttentionBridge),  # Output projection
                    "mlp": ("intermediate", MLPBridge),  # Full MLP module
                    "mlp.fc1": ("intermediate.dense", MLPBridge),  # First linear layer
                    "mlp.fc2": ("output.dense", MLPBridge),  # Second linear layer
                },
            ),
            "unembed": ("cls.predictions", UnembeddingBridge),  # Language model head
        }
