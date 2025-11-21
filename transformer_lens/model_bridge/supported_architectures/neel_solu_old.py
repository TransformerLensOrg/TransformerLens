"""Neel Solu Old architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class NeelSoluOldArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Neel's SOLU models (old style)."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Neel SOLU old-style architecture adapter.

        Args:
            cfg: The configuration object.
        """
        self.default_config: dict[str, Any] = {}
        super().__init__(cfg)

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model n_head d_head -> n_head d_model d_head"
                ),
                source_key="blocks.{i}.attn.W_Q",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model n_head d_head -> n_head d_model d_head"
                ),
                source_key="blocks.{i}.attn.W_K",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model n_head d_head -> n_head d_model d_head"
                ),
                source_key="blocks.{i}.attn.W_V",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "n_head d_head d_model -> n_head d_head d_model"
                ),
                source_key="blocks.{i}.attn.W_O",
            ),
        }
        self.component_mapping = {
            "embed": EmbeddingBridge(name="wte"),
            "pos_embed": PosEmbedBridge(name="wpe"),
            "blocks": BlockBridge(
                name="blocks",
                submodules={
                    "ln1": NormalizationBridge(name="ln1", config=self.cfg),
                    "attn": AttentionBridge(name="attn", config=self.cfg),
                    "ln2": NormalizationBridge(name="ln2", config=self.cfg),
                    "mlp": MLPBridge(name="mlp"),
                },
            ),
            "ln_final": NormalizationBridge(name="ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="unembed"),
        }


def convert_neel_solu_old_weights(state_dict: dict, cfg: Any):
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
