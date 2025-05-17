"""Architecture conversion base class.

This module contains the base class for architecture conversions.
"""

from abc import ABC
from typing import Any

import torch
from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.helpers.merge_quantiziation_fields import (
    merge_quantization_fields,
)
from transformer_lens.architecture_adapter.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.architecture_adapter.generalized_components.embedding import (
    EmbeddingBridge,
)
from transformer_lens.architecture_adapter.generalized_components.layer_norm import (
    LayerNormBridge,
)
from transformer_lens.architecture_adapter.generalized_components.mlp import MLPBridge
from transformer_lens.architecture_adapter.generalized_components.unembedding import (
    UnembeddingBridge,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from .conversion_steps.types import FIELD_SET


class ArchitectureConversion(ABC):
    """Base class for architecture conversions."""

    def __init__(self, cfg: HookedTransformerConfig):
        """Initialize the architecture conversion.

        Args:
            cfg: The config to use for the conversion.
        """
        self.cfg = cfg
        self.field_set = None

    def enable_quantiziation(
        self, cfg: HookedTransformerConfig, quantiziation_fields: FIELD_SET
    ) -> None:
        if cfg.load_in_4bit:
            self.field_set = merge_quantization_fields(self.field_set, quantiziation_fields)

    def get_component(self, model: PreTrainedModel, name: str) -> Any:
        """Get a component from the model using the field set mapping.

        This method maps HookedTransformer component names to the underlying model's structure
        using the field set mapping. It wraps the original components with bridge components
        that provide standardized hook points.

        Args:
            model: The model to get the component from.
            name: The name of the component to get.

        Returns:
            The requested component, wrapped in a bridge component if applicable.
        """
        if self.field_set is None:
            raise ValueError("field_set must be set before calling get_component")
            
        # Get the original component
        original_component = self.field_set.get_component(model, name)
        
        # Wrap with appropriate bridge component based on name
        if name == "embed":
            return EmbeddingBridge(original_component, name)
        elif name == "unembed":
            return UnembeddingBridge(original_component, name)
        elif name.endswith(".ln1") or name.endswith(".ln2") or name == "ln_final":
            return LayerNormBridge(original_component, name)
        elif name.endswith(".attn"):
            return AttentionBridge(original_component, name)
        elif name.endswith(".mlp"):
            return MLPBridge(original_component, name)
            
        # Return original component for other cases
        return original_component

    def convert_weights(self, hf_model: PreTrainedModel) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        if self.field_set is None:
            raise ValueError("field_set must be set before calling convert_weights")
        return self.field_set.convert(input_value=hf_model)
