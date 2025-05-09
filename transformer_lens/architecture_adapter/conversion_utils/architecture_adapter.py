"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.generalized_components import (
    AttentionBridge,
    MLPBridge,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class ArchitectureAdapter(ABC):
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

    @abstractmethod
    def get_component(self, model: PreTrainedModel, name: str) -> Any:
        """Get a component from the model.
        
        Args:
            model: The model to get the component from
            name: The name of the component to get
            
        Returns:
            The requested component
        """
        pass

    def _get_component_type(self, name: str) -> str:
        """Get the type information for a component.
        
        Args:
            name: The name of the component in the HuggingFace model
            
        Returns:
            A string describing the component's type and shape
        """
        try:
            # Navigate through the model's structure using the component name
            parts = name.split(".")
            component = self.model
            for part in parts:
                # Handle array indexing in the name (e.g., "h.0" -> "h[0]")
                if part.isdigit():
                    component = component[int(part)]
                else:
                    component = getattr(component, part)

            # Get the component's type and shape
            if isinstance(component, torch.Tensor):
                shape_str = "×".join(str(s) for s in component.shape)
                dtype_str = str(component.dtype).replace("torch.", "")
                return f"Tensor({shape_str}, {dtype_str})"
            elif isinstance(component, nn.Module):
                # For bridge components, show both the wrapper and original class
                if hasattr(component, 'original_component'):
                    orig_class = component.original_component.__class__.__name__
                    wrapper_class = component.__class__.__name__
                    return f"{wrapper_class}({orig_class})"
                return component.__class__.__name__
            else:
                return type(component).__name__
        except (AttributeError, IndexError):
            return "Unknown"

    def _wrap_component(self, current: nn.Module, name: str) -> nn.Module:
        """Wrap a component with its bridge if needed.
        
        Args:
            current: The component to wrap
            name: The name of the component
            
        Returns:
            The wrapped component
        """
        if name.endswith(".attn"):
            return AttentionBridge(current, name)
        elif name.endswith(".mlp"):
            return MLPBridge(current, name)
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