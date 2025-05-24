"""OPT architecture adapter."""

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


class OPTArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OPT models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the OPT architecture adapter.

        Args:
            cfg: The HookedTransformer configuration.
        """
        super().__init__(cfg)

        self.conversion_rules = WeightConversionSet(
            {
                "embed.W_E": "model.decoder.embed_tokens.weight",
                "blocks.{i}.ln1.w": "model.decoder.layers.{i}.self_attn_layer_norm.weight",
                "blocks.{i}.ln1.b": "model.decoder.layers.{i}.self_attn_layer_norm.bias",
                "blocks.{i}.ln2.w": "model.decoder.layers.{i}.final_layer_norm.weight",
                "blocks.{i}.ln2.b": "model.decoder.layers.{i}.final_layer_norm.bias",
                "blocks.{i}.attn.W_Q": (
                    "model.decoder.layers.{i}.self_attn.q_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_K": (
                    "model.decoder.layers.{i}.self_attn.k_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.W_V": (
                    "model.decoder.layers.{i}.self_attn.v_proj.weight",
                    RearrangeWeightConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "model.decoder.layers.{i}.self_attn.q_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "model.decoder.layers.{i}.self_attn.k_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "model.decoder.layers.{i}.self_attn.v_proj.bias",
                    RearrangeWeightConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "model.decoder.layers.{i}.self_attn.out_proj.weight",
                    RearrangeWeightConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "model.decoder.layers.{i}.self_attn.out_proj.bias",
                "blocks.{i}.mlp.W_in": "model.decoder.layers.{i}.fc1.weight",
                "blocks.{i}.mlp.b_in": "model.decoder.layers.{i}.fc1.bias",
                "blocks.{i}.mlp.W_out": "model.decoder.layers.{i}.fc2.weight",
                "blocks.{i}.mlp.b_out": "model.decoder.layers.{i}.fc2.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
                "ln_final.w": "model.decoder.final_layer_norm.weight",
                "ln_final.b": "model.decoder.final_layer_norm.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("model.decoder.embed_tokens", EmbeddingBridge),
            "pos_embed": ("model.decoder.embed_positions", EmbeddingBridge),
            "blocks": (
                "model.decoder.layers",
                {
                    "ln1": ("self_attn_layer_norm", LayerNormBridge),
                    "ln2": ("final_layer_norm", LayerNormBridge),
                    "attn": ("self_attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            ),
            "ln_final": ("model.decoder.final_layer_norm", LayerNormBridge),
            "unembed": ("lm_head", UnembeddingBridge),
        }
