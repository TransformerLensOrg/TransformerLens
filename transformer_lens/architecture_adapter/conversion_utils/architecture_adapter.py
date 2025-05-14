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

RemoteModel: TypeAlias = nn.Module
RemoteComponent: TypeAlias = nn.Module


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

    def get_remote_component(self, model: RemoteModel, path: RemotePath) -> RemoteComponent:
        """Get a component from the remote model using a dot-separated path.
        
        Args:
            model: The remote model to extract the component from
            path: The dot-separated path to the component (e.g. "model.layers.0.ln1")
            
        Returns:
            The component at the specified path
            
        Raises:
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed
            ValueError: If the path is empty or invalid
            
        Examples:
            >>> adapter.get_remote_component(model, "model.embed_tokens")
            <Embedding>
            >>> adapter.get_remote_component(model, "model.layers.0")
            <TransformerBlock>
            >>> adapter.get_remote_component(model, "model.layers.0.ln1")
            <LayerNorm>
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        parts = path.split(".")
        current: Any = model
        
        for part in parts:
            if not part:
                raise ValueError(f"Invalid path segment in {path}")
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
                
        if not isinstance(current, nn.Module):
            raise ValueError(f"Component at path {path} is not a nn.Module")
            
        return current

    def translate_transformer_lens_path(self, path: TransformerLensPath) -> RemotePath:
        """Translate a TransformerLens path to its corresponding Remote path.
        
        Args:
            path: The TransformerLens path to translate (e.g. "blocks.0.ln1")
            
        Returns:
            The corresponding Remote path (e.g. "model.layers.0.input_layernorm")
            
        Raises:
            ValueError: If the component mapping is not set or if the path is invalid
            
        Examples:
            >>> adapter.translate_transformer_lens_path("embed")
            "model.embed_tokens"
            >>> adapter.translate_transformer_lens_path("blocks.0")
            "model.layers.0"
            >>> adapter.translate_transformer_lens_path("blocks.0.ln1")
            "model.layers.0.input_layernorm"
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling translate_transformer_lens_path")
            
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")
            
        # First part should be a top-level component
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")
            
        return self._resolve_component_path(parts, self.component_mapping[parts[0]])

    def _resolve_component_path(self, parts: list[str], mapping: RemotePath | tuple[RemotePath, ComponentLayer]) -> RemotePath:
        """Recursively resolve a component path to its remote path.
        
        Args:
            parts: List of path components to resolve
            mapping: Current level of component mapping
            
        Returns:
            The resolved remote path
            
        Raises:
            ValueError: If the path is invalid or component not found
        """
        if not parts:
            raise ValueError("Empty path")
            
        # Handle tuple case (base_path, sub_mapping)
        if isinstance(mapping, tuple):
            base_path, sub_mapping = mapping
            # If we're at a leaf node (just the index)
            if len(parts) == 1:
                if not parts[0].isdigit():
                    raise ValueError(f"Expected index, got {parts[0]}")
                return f"{base_path}.{parts[0]}"
            # Otherwise, continue with the sub_mapping
            if not parts[0].isdigit():
                raise ValueError(f"Expected index, got {parts[0]}")
            idx = parts[0]
            return f"{base_path}.{idx}.{self._resolve_component_path(parts[1:], sub_mapping)}"
            
        # Handle string case (direct path)
        if len(parts) == 1:
            return mapping
        return f"{mapping}.{'.'.join(parts[1:])}"

    def get_component(self, model: RemoteModel, path: TransformerLensPath) -> RemoteComponent:
        """Get a component from the model using the component_mapping.
        
        Args:
            model: The model to extract components from
            path: The path of the component to get, as defined in component_mapping
            
        Returns:
            The requested component from the model
            
        Raises:
            ValueError: If component_mapping is not set or if the component is not found
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed
            
        Examples:
            >>> adapter.get_component(model, "embed")
            <Embedding>
            >>> adapter.get_component(model, "blocks.0")
            <TransformerBlock>
            >>> adapter.get_component(model, "blocks.0.ln1")
            <LayerNorm>
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")
            
        # Get the remote path and then get the component
        remote_path = self.translate_transformer_lens_path(path)
        return self.get_remote_component(model, remote_path)

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