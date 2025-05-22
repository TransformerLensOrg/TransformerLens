"""Neel Solu Old architecture adapter."""

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


class NeelSoluOldArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Neel Solu Old models."""

    def __init__(self, cfg: HookedTransformerConfig) -> None:
        """Initialize the Neel Solu Old architecture adapter.

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
            "embed": ("transformer.wte", EmbeddingBridge),  # Word token embeddings
            "blocks": (
                "transformer.h",  # Base path for blocks
                {
                    "ln1": ("ln_1", LayerNormBridge),  # Pre-attention layer norm
                    "ln2": ("ln_2", LayerNormBridge),  # Pre-MLP layer norm
                    "attn": ("attn", AttentionBridge),  # Full attention module
                    "attn.c_attn": ("attn.c_attn", AttentionBridge),  # Combined QKV projection
                    "attn.c_proj": ("attn.c_proj", AttentionBridge),  # Output projection
                    "mlp": ("mlp", MLPBridge),  # Full MLP module
                    "mlp.c_fc": ("mlp.c_fc", MLPBridge),  # First linear layer
                    "mlp.c_proj": ("mlp.c_proj", MLPBridge),  # Second linear layer
                },
            ),
            "ln_final": ("transformer.ln_f", LayerNormBridge),  # Final layer norm
            "unembed": ("lm_head", UnembeddingBridge),  # Language model head
        }

def convert_neel_solu_old_weights(state_dict: dict, cfg: HookedTransformerConfig):
    """
    Converts the weights of my old SoLU models to the HookedTransformer format.
    Takes as input a state dict, *not* a model object.

    There are a bunch of dumb bugs in the original code, sorry!

    Models 1L, 2L, 4L and 6L have left facing weights (ie, weights have shape
    [dim_out, dim_in]) while HookedTransformer does right facing (ie [dim_in,
    dim_out]).

    8L has *just* a left facing W_pos, the rest right facing.

    And some models were trained with
    """
    # Early models have left facing W_pos
    reverse_pos = cfg.n_layers <= 8

    # Models prior to 8L have left facing everything (8L has JUST left facing W_pos - sorry! Stupid bug)
    reverse_weights = cfg.n_layers <= 6

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm", "ln")
        if k.startswith("ln."):
            k = k.replace("ln.", "ln_final.")
        new_state_dict[k] = v

    if reverse_pos:
        new_state_dict["pos_embed.W_pos"] = new_state_dict["pos_embed.W_pos"].T
    if reverse_weights:
        for k, v in new_state_dict.items():
            if "W_" in k and "W_pos" not in k:
                new_state_dict[k] = v.transpose(-2, -1)
    return new_state_dict
