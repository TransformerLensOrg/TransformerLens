"""Residual Decomposition.

TODO: Motivation for why residual decomposition is useful (DLA, logit lens...)

TODO: Describe the high level principle of being able to break things down, as they are linear.
"""

from typing import Literal, Optional, List, Tuple
from transformer_lens import ActivationCache, HookedTransformerConfig
from transformer_lens.utils import Slice
from jaxtyping import Float
from torch import Tensor
from enum import Enum


class DecompositionLabel:
    destination_layer: int
    destination_layer_type: Literal["mlp", "attention", "embed", "unembed"]
    destination_head: Optional[int]
    source_layer: Optional[int]
    source_layer_type: Optional[Literal["mlp", "attention", "embed", "unembed"]]
    source_head: Optional[int]


DecompositionLabels = List[DecompositionLabel]

# For layer in layers
#  For head in heads
#    For source token in sorce tokens
#      For source component in source
#        Get MLPs


def decompose_layers(
    cache: ActivationCache,
    model_config: HookedTransformerConfig,
    batch_slice: Optional[Slice] = None,
    pos_slice: Optional[Slice] = None,
    layer_slice: Optional[Slice] = None,
    component_types: Optional[List[Literal["mlp", "attention", "embed", "unembed"]]] = [
        "mlp",
        "attention",
        "embed",
        "unembed",
    ],
) -> Tuple[DecompositionLabels, Float[Tensor, "component batch pos d_model"]]:
    """Decompose the residual stream into the output of each layer"""
    pass


def decompose_mlp_neurons(cache: ActivationCache, layer: int):
    """Decompose the residual stream into MLP Neurons"""
    pass


def decompose_attention_source_tokens(cache: ActivationCache, layer: int, head: int):
    """Decompose the residual stream into source tokens

    Attention head -> source tokens

    Have to get the attention pattern from the cache (i.e.)
    """
    pass


def decompose_attention_source_components(
    cache: ActivationCache, source_position: int, source_layer: int
):
    """Decompose the residual stream into source components"""
    pass
