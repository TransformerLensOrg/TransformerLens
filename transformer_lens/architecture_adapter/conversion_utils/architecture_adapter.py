"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from transformer_lens.architecture_adapter.generalized_components import (
    AttentionBridge,
    MLPBridge,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

# Type aliases for paths
TransformerLensPath: TypeAlias = str  # Path in TransformerLens format (e.g. "blocks.0.attn")
RemotePath: TypeAlias = str  # Path in the remote model format (e.g. "transformer.h.0.attn")

# Component mapping types
ComponentLayer: TypeAlias = dict[TransformerLensPath, RemotePath]  # Maps TransformerLens components to remote components
BlockMapping: TypeAlias = tuple[RemotePath, ComponentLayer]  # Maps a block and its components
ComponentMapping: TypeAlias = dict[TransformerLensPath, RemotePath | BlockMapping]  # Complete component mapping


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
        self.component_mapping: ComponentMapping | None = None

    def get_component_path(self, name: str) -> str:
        """Get the path to a component in the model.
        
        Args:
            name: The name of the component to get the path for
            
        Returns:
            The path to the component in the model
            
        Raises:
            ValueError: If the component mapping is not set or the component is not found
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component_path")
            
        if name not in self.component_mapping:
            raise ValueError(f"Component {name} not found in component mapping")
            
        return self.component_mapping[name]

    def get_block_component_path(self, block_idx: int, component_name: str) -> str:
        """Get the path to a component within a block.
        
        Args:
            block_idx: The index of the block
            component_name: The name of the component within the block
            
        Returns:
            The path to the component in the model
            
        Raises:
            ValueError: If the component mapping is not set or the component is not found
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_block_component_path")
            
        if "blocks" not in self.component_mapping:
            raise ValueError("No blocks found in component mapping")
            
        blocks_info = self.component_mapping["blocks"]
        if not isinstance(blocks_info, tuple) or len(blocks_info) != 2:
            raise ValueError("Invalid blocks mapping format")
            
        base_path, block_components = blocks_info
        if component_name not in block_components:
            raise ValueError(f"Component {component_name} not found in block components")
            
        return f"{base_path}.{block_idx}.{block_components[component_name]}"

    def get_component_mapping(self) -> ComponentMapping:
        """Get the full component mapping.
        
        Returns:
            The component mapping dictionary
            
        Raises:
            ValueError: If the component mapping is not set
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component_mapping")
        return self.component_mapping


    def get_component(self, model: Any, path: TransformerLensPath) -> nn.Module:
        """Get a component from the model using the component_mapping.
        
        Args:
            model: The model to extract components from
            path: The path of the component to get, as defined in component_mapping
            
        Returns:
            The requested component from the model
            
        Raises:
            ValueError: If component_mapping is not set or if the component is not found
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")
            
        parts = path.split(".")
        current = model
        
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
                
        return current

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