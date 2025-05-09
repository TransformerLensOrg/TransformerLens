"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from typing import Any

import torch
from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.generalized_components import (
    GeneralizedAttention,
    GeneralizedMLP,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class ArchitectureAdapter:
    """Base class for architecture adapters.
    
    This class provides the interface for adapting between different model architectures.
    It handles both component mapping (for accessing model parts) and weight conversion
    (for initializing weights from one format to another).
    """

    def __init__(self, cfg: HookedTransformerConfig):
        """Initialize the architecture adapter.

        Args:
            cfg: The config to use for the adapter.
        """
        self.cfg = cfg
        self.conversion_rules = None
        self.component_mapping = None

    def get_component(self, model: PreTrainedModel, name: str) -> Any:
        """Get a component from the model using the component mapping.

        This method maps HookedTransformer component names to the underlying model's structure
        using the component mapping dictionary. It handles nested structures recursively.

        Args:
            model: The model to get the component from.
            name: The name of the component to get.

        Returns:
            The requested component.

        Raises:
            ValueError: If component_mapping is not set or if the component path is invalid.
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")

        def resolve_path(path_parts: list[str], mapping: dict[str, Any] | tuple[str, dict[str, str]]) -> str:
            """Recursively resolve a component path to its underlying model path.
            
            Args:
                path_parts: List of path components.
                mapping: Current level of component mapping.
            
            Returns:
                The resolved path in the underlying model.
            """
            if not path_parts:
                raise ValueError("Empty path")

            # Handle tuple case (base_path, sub_mapping)
            if isinstance(mapping, tuple):
                base_path, sub_mapping = mapping
                # If we're at a leaf node (just the block index)
                if len(path_parts) == 1:
                    if not path_parts[0].isdigit():
                        raise ValueError(f"Expected layer index, got {path_parts[0]}")
                    return f"{base_path}.{path_parts[0]}"
                # Otherwise, continue with the sub_mapping
                if not path_parts[0].isdigit():
                    raise ValueError(f"Expected layer index, got {path_parts[0]}")
                layer_idx = path_parts[0]
                return f"{base_path}.{layer_idx}.{resolve_path(path_parts[1:], sub_mapping)}"

            # Handle dictionary case
            current = path_parts[0]
            if current not in mapping:
                raise ValueError(f"Unknown component: {current}")
            
            value = mapping[current]
            # If this is a leaf node (string path)
            if isinstance(value, str):
                if len(path_parts) == 1:
                    return value
                # If there are more parts, append them to the path
                return f"{value}.{'.'.join(path_parts[1:])}"
            # If this is a nested structure, recurse
            return resolve_path(path_parts[1:], value)

        # Parse the component path and resolve it
        parts = name.split(".")
        component_path = resolve_path(parts, self.component_mapping)

        # Navigate through the model to get the component
        current = model
        for part in component_path.split("."):
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)

        # Wrap with appropriate generalized component based on name
        if name.endswith(".attn"):
            return GeneralizedAttention(current, name)
        elif name.endswith(".mlp"):
            return GeneralizedMLP(current, name)
            
        # Return original component for other cases
        return current

    def convert_weights(self, hf_model: PreTrainedModel) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        if self.conversion_rules is None:
            raise ValueError("conversion_rules must be set before calling convert_weights")
        return self.conversion_rules.convert(input_value=hf_model) 