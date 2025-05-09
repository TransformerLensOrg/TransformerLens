"""Bridge between HookedTransformer and underlying model architectures.

This module provides a pure adapter layer that maps HookedTransformer's component access patterns
to the underlying model's structure using architecture adapters.
"""

from typing import Any

import torch
from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)


class BlockComponentProxy:
    """Proxy class for block component access."""

    def __init__(self, bridge: "TransformerBridge", block_idx: int):
        """Initialize the block component proxy.

        Args:
            bridge: The bridge instance.
            block_idx: The block index.
        """
        self.bridge = bridge
        self.block_idx = block_idx

    def __getattr__(self, name: str) -> Any:
        """Get a block component by name.

        Args:
            name: The component name.

        Returns:
            The requested component.
        """
        return self.bridge.architecture_adapter.get_component(
            self.bridge.model, f"blocks.{self.block_idx}.{name}"
        )


class BlockProxy:
    """Proxy class for block access that handles array indexing."""

    def __init__(self, bridge: "TransformerBridge"):
        """Initialize the block proxy.

        Args:
            bridge: The bridge instance.
        """
        self.bridge = bridge

    def __getitem__(self, idx: int) -> BlockComponentProxy:
        """Get a block by index.

        Args:
            idx: The block index.

        Returns:
            A proxy for accessing the block's components.
        """
        return BlockComponentProxy(self.bridge, idx)


class TransformerBridge:
    """Bridge between HookedTransformer and underlying model architectures."""

    def __init__(
        self,
        model: PreTrainedModel,
        architecture_adapter: ArchitectureConversion,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the transformer bridge.

        Args:
            model: The underlying model to bridge to.
            architecture_adapter: The architecture adapter to use for mapping between models.
            device: The device to put the model on.
            dtype: The dtype to use for the model.
        """
        self.model = model
        self.architecture_adapter = architecture_adapter
        self.device = device
        self.dtype = dtype
        self._blocks = BlockProxy(self)

    def __getattr__(self, name: str) -> Any:
        """Get a component from the model using the architecture adapter.

        This method allows HookedTransformer to access components using its own naming scheme,
        which are then mapped to the underlying model's structure using the architecture adapter.

        Args:
            name: The name of the component to get.

        Returns:
            The requested component.
        """
        if name == "blocks":
            return self._blocks
        return self.architecture_adapter.get_component(self.model, name)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "TransformerBridge":
        """Move the model to the specified device and dtype.

        Args:
            device: The device to move the model to.
            dtype: The dtype to convert the model to.

        Returns:
            self: The bridge instance.
        """
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the model.

        This method delegates the forward pass to the underlying model.

        Args:
            *args: Positional arguments to pass to the model.
            **kwargs: Keyword arguments to pass to the model.

        Returns:
            The model's output.
        """
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate text using the model.

        This method delegates text generation to the underlying model.

        Args:
            *args: Positional arguments to pass to the model's generate method.
            **kwargs: Keyword arguments to pass to the model's generate method.

        Returns:
            The generated output.
        """
        return self.model.generate(*args, **kwargs)

    def _get_component_type(self, name: str) -> str:
        """Get the type information for a component.

        Args:
            name: The name of the component in the HuggingFace model.

        Returns:
            A string describing the component's type and shape.
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
            elif isinstance(component, torch.nn.Module):
                return component.__class__.__name__
            else:
                return type(component).__name__
        except (AttributeError, IndexError):
            return "Unknown"

    def _format_field_mapping(self) -> list[str]:
        """Format the field mapping into a readable structure.

        Returns:
            A list of strings representing the field mapping structure.
        """
        component_mapping = getattr(self.architecture_adapter, "component_mapping", None)
        if not isinstance(component_mapping, dict):
            return ["    Component Mapping: Not available"]

        lines = ["    Component Mapping:"]

        # Format top-level components
        for tl_name, hf_name in component_mapping.items():
            if tl_name == "blocks":
                continue  # Handle blocks separately
            component_type = self._get_component_type(hf_name)
            lines.append(f"        {tl_name} -> {hf_name} ({component_type})")

        # Format blocks structure
        if "blocks" in component_mapping:
            base_path, sub_mapping = component_mapping["blocks"]
            lines.append(f"        blocks: (base_path: {base_path})")
            for tl_name, hf_name in sub_mapping.items():
                # Construct full path for type lookup
                full_path = f"{base_path}.0.{hf_name}"  # Use layer 0 for type lookup
                component_type = self._get_component_type(full_path)
                lines.append(f"            {tl_name} -> {hf_name} ({component_type})")

        return lines

    def __repr__(self) -> str:
        """Get a string representation of the bridge.

        Returns:
            A detailed string representation showing the bridge's components and configuration.
        """
        model_config = self.model.config
        adapter_config = self.architecture_adapter.cfg

        # Build the representation string
        lines = [
            "TransformerBridge(",
            f"    Model: {getattr(model_config, 'name_or_path', 'Unknown')}",
            f"    Architecture: {model_config.architectures[0] if model_config.architectures else 'Unknown'}",
            f"    Device: {self.device}",
            f"    Dtype: {self.dtype}",
            "    Model Config:",
            f"        Hidden Size: {model_config.hidden_size}",
            f"        Num Layers: {model_config.num_hidden_layers}",
            f"        Num Attention Heads: {model_config.num_attention_heads}",
            f"        Vocab Size: {model_config.vocab_size}",
            "    Adapter Config:",
            f"        D Model: {adapter_config.d_model}",
            f"        N Layers: {adapter_config.n_layers}",
            f"        N Heads: {adapter_config.n_heads}",
            f"        D Head: {adapter_config.d_head}",
            f"        D MLP: {adapter_config.d_mlp}",
            f"        D Vocab: {adapter_config.d_vocab}",
        ]

        # Add the component mapping
        lines.extend(self._format_field_mapping())
        lines.append(")")

        return "\n".join(lines) 