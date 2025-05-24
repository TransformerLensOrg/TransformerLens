"""Phi architecture adapter."""

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
                "embed.W_E": "transformer.wte.weight",
                "blocks.{i}.ln1.w": "transformer.h.{i}.ln_1.weight",
                "blocks.{i}.ln1.b": "transformer.h.{i}.ln_1.bias",
                "blocks.{i}.ln2.w": "transformer.h.{i}.ln_2.weight",
                "blocks.{i}.ln2.b": "transformer.h.{i}.ln_2.bias",
                "blocks.{i}.attn.W_Q": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.W_K": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.W_V": (
                    "transformer.h.{i}.attn.c_attn.weight",
                    RearrangeWeightConversion(
                        "d_model (3 n_head d_head) -> 3 n_head d_head d_model"
                    ),
                ),
                "blocks.{i}.attn.b_Q": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "transformer.h.{i}.attn.c_attn.bias",
                    RearrangeWeightConversion("(3 n_head d_head) -> 3 n_head d_head"),
                ),
                "blocks.{i}.attn.W_O": (
                    "transformer.h.{i}.attn.c_proj.weight",
                    RearrangeWeightConversion("(n_head d_head) d_model -> n_head d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "transformer.h.{i}.attn.c_proj.bias",
                "blocks.{i}.mlp.W_in": "transformer.h.{i}.mlp.c_fc.weight",
                "blocks.{i}.mlp.b_in": "transformer.h.{i}.mlp.c_fc.bias",
                "blocks.{i}.mlp.W_out": "transformer.h.{i}.mlp.c_proj.weight",
                "blocks.{i}.mlp.b_out": "transformer.h.{i}.mlp.c_proj.bias",
                "ln_final.w": "transformer.ln_f.weight",
                "ln_final.b": "transformer.ln_f.bias",
                "unembed.W_U": "lm_head.weight",
                "unembed.b_U": "lm_head.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),
            "blocks": (
                "model.layers",
                {
                    "ln1": ("input_layernorm", LayerNormBridge),
                    "ln2": ("input_layernorm", LayerNormBridge),
                    "attn": ("self_attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            ),
            "ln_final": ("model.final_layernorm", LayerNormBridge),
            "unembed": ("lm_head", UnembeddingBridge),
        }
