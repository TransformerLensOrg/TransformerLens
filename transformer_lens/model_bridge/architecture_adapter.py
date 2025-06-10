"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from typing import Any

import torch
from transformers.modeling_utils import PreTrainedModel

from transformer_lens.model_bridge.conversion_utils.conversion_steps import (
    WeightConversionSet,
)
from transformer_lens.model_bridge.types import (
    BlockMapping,
    ComponentMapping,
    RemoteComponent,
    RemoteImport,
    RemoteModel,
    RemotePath,
    TransformerLensPath,
)


class ArchitectureAdapter:
    """Base class for architecture adapters.

    This class provides the interface for adapting between different model architectures.
    It handles both component mapping (for accessing model parts) and weight conversion
    (for initializing weights from one format to another).
    """

    def __init__(self, user_cfg: Any) -> None:
        """Initialize the architecture adapter.

        Args:
            user_cfg: The user-provided configuration object.
        """
        self.user_cfg = user_cfg
        self.default_cfg: dict[str, Any] = {}
        self.component_mapping: ComponentMapping | None = None
        self.conversion_rules: WeightConversionSet | None = None

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
        """Get a component from a remote model by its path.

        This method should be overridden by subclasses to provide the logic for
        accessing components in a specific model architecture.

        Args:
            model: The remote model
            path: The path to the component in the remote model's format

        Returns:
            The component (e.g., a PyTorch module)

        Raises:
            AttributeError: If a component in the path doesn't exist
            IndexError: If an invalid index is accessed
            ValueError: If the path is empty or invalid

        Examples:
            Get an embedding component:

            >>> # adapter.get_remote_component(model, "model.embed_tokens")
            >>> # <Embedding>

            Get a transformer block:

            >>> # adapter.get_remote_component(model, "model.layers.0")
            >>> # <TransformerBlock>

            Get a layer norm component:

            >>> # adapter.get_remote_component(model, "model.layers.0.ln1")
            >>> # <LayerNorm>
        """
        current = model
        for part in path.split("."):
            if part.isdigit():
                current = current[int(part)]  # type: ignore[index]
            else:
                current = getattr(current, part)
        return current

    def translate_transformer_lens_path(
        self, path: TransformerLensPath, last_component_only: bool = False
    ) -> RemotePath:
        """Translate a TransformerLens path to its corresponding Remote path.

        Args:
            path: The TransformerLens path to translate (e.g. "blocks.0.ln1")
            last_component_only: If True, only return the last component of the path (e.g. "input_layernorm")

        Returns:
            The corresponding Remote path (e.g. "model.layers.0.input_layernorm" or just "input_layernorm")

        Raises:
            ValueError: If the component mapping is not set or if the path is invalid
            KeyError: If the path is not found in the component mapping.
        """
        component_mapping = self.get_component_mapping()
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # First part should be a top-level component
        if parts[0] not in component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        full_path = self._resolve_component_path(parts[1:], component_mapping[parts[0]])

        if last_component_only:
            # Split the path and return only the last component
            return full_path.split(".")[-1]

        return full_path

    def _resolve_component_path(
        self, parts: list[str], mapping: RemoteImport | BlockMapping
    ) -> RemotePath:
        """Recursively resolve a component path to its remote path.

        Args:
            parts: List of path components to resolve
            mapping: Current level of component mapping (either RemoteImport or BlockMapping)

        Returns:
            The resolved remote path

        Raises:
            ValueError: If the path is invalid or component not found
        """
        if not parts:
            # For both RemoteImport and BlockMapping, return the base path
            return mapping[0]  # Return the base path (first element of tuple)

        if len(mapping) == 2:
            # This is a RemoteImport (path, bridge_type)
            base_path, _ = mapping
            return f"{base_path}.{'.'.join(parts)}"
        elif len(mapping) == 3:
            # This is a BlockMapping (path, bridge_type, sub_mapping)
            base_path, _, sub_mapping = mapping
        else:
            raise ValueError(f"Invalid mapping structure: {mapping}")

        # Handle BlockMapping case
        idx = parts[0]
        if not idx.isdigit():
            raise ValueError(f"Expected index, got {idx}")
        if len(parts) == 1:
            return f"{base_path}.{idx}"

        # If next part is a subcomponent, look it up in sub_mapping
        sub_name = parts[1]
        if sub_name not in sub_mapping:
            raise ValueError(f"Component {sub_name} not found in blocks components")
        sub_map = sub_mapping[sub_name]

        # If there are more parts, recurse into sub_map
        if len(parts) > 2:
            return self._resolve_component_path(parts[2:], sub_map)
        return f"{base_path}.{idx}.{sub_map[0]}"  # Use first element (path) from RemoteImport

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
            Get an embedding component:

            >>> # adapter.get_component(model, "embed")
            >>> # <Embedding>

            Get a transformer block:

            >>> # adapter.get_component(model, "blocks.0")
            >>> # <TransformerBlock>

            Get a layer norm component:

            >>> # adapter.get_component(model, "blocks.0.ln1")
            >>> # <LayerNorm>
        """
        if self.component_mapping is None:
            raise ValueError("component_mapping must be set before calling get_component")

        # Get the remote path and then get the component
        remote_path = self.translate_transformer_lens_path(path)
        return self.get_remote_component(model, remote_path)

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

    def get_remote_path_and_type(self, tl_path: str) -> RemoteImport:
        """Get the remote path and type for a given TransformerLens path.

        Args:
            tl_path: The TransformerLens path (e.g., "blocks.0.attn")

        Returns:
            A tuple of (remote_path, remote_type)

        Raises:
            KeyError: If the path is not found in the component mapping.
        """
        current_mapping = self.get_component_mapping()
        path_parts = tl_path.split(".")
        for i, part in enumerate(path_parts):
            assert isinstance(current_mapping, dict)
            if part not in current_mapping:
                raise KeyError(f"Path {tl_path} not found in component_mapping.")
            entry = current_mapping[part]
            if isinstance(entry, tuple) and len(entry) == 3:  # BlockMapping
                if i == len(path_parts) - 1:
                    # We are at the block itself
                    return entry[0], entry[1]
                else:
                    # We are descending into a block
                    current_mapping = entry[2]
                    assert isinstance(current_mapping, dict)
            elif isinstance(entry, tuple) and len(entry) == 2:  # RemoteImport
                if i != len(path_parts) - 1:
                    raise KeyError(f"Path {tl_path} is too long.")
                return entry
            else:
                raise TypeError(f"Invalid entry in component_mapping: {entry}")
        raise KeyError(f"Path {tl_path} not found in component_mapping.")
