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
    ComponentMapping,
    RemoteComponent,
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
        # Ensure d_mlp is set if intermediate_size is present
        if not hasattr(self.user_cfg, "d_mlp") and hasattr(self.user_cfg, "intermediate_size"):
            self.user_cfg.d_mlp = self.user_cfg.intermediate_size
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

        # In the new system, we get the bridge component from the mapping
        # and use its name attribute to get the remote component
        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the component at the bridge's remote path
            return self.get_remote_component(model, bridge_component.name)

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if parts[0] == "blocks" and len(parts) >= 2:
            # Handle blocks indexing
            block_index = parts[1]
            if not block_index.isdigit():
                raise ValueError(f"Expected block index, got {block_index}")

            # Get the block container
            block_container = self.get_remote_component(model, bridge_component.name)
            block = block_container[int(block_index)]

            if len(parts) == 2:
                # Just return the block
                return block
            else:
                # Get subcomponent from the block
                subcomponent_path = ".".join(parts[2:])
                current = block
                for part in subcomponent_path.split("."):
                    current = getattr(current, part)
                return current

        # For other nested paths, navigate through the remote model
        remote_path = bridge_component.name
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        return self.get_remote_component(model, remote_path)

    def translate_transformer_lens_path(
        self, path: TransformerLensPath, last_component_only: bool = False
    ) -> RemotePath:
        """Translate a TransformerLens path to a remote model path.

        Args:
            path: The TransformerLens path to translate
            last_component_only: If True, return only the last component of the path

        Returns:
            The corresponding remote model path

        Raises:
            ValueError: If the path is not found in the component mapping

        Examples:
            >>> adapter.translate_transformer_lens_path("embed")
            "model.embed_tokens"
            >>> adapter.translate_transformer_lens_path("blocks.0.ln1")
            "model.layers.0.input_layernorm"
            >>> adapter.translate_transformer_lens_path("embed", last_component_only=True)
            "embed_tokens"
        """
        if self.component_mapping is None:
            raise ValueError(
                "component_mapping must be set before calling translate_transformer_lens_path"
            )

        parts = path.split(".")
        if not parts:
            raise ValueError("Empty path")

        # Get the top-level component from the mapping
        if parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the bridge's remote path
            remote_path = bridge_component.name
            if last_component_only:
                return remote_path.split(".")[-1]
            return remote_path

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if parts[0] == "blocks" and len(parts) >= 2:
            # Handle blocks indexing
            block_index = parts[1]
            if not block_index.isdigit():
                raise ValueError(f"Expected block index, got {block_index}")

            # Get the base blocks path
            blocks_path = bridge_component.name

            if len(parts) == 2:
                # Just return the indexed block path
                remote_path = f"{blocks_path}.{block_index}"
                if last_component_only:
                    return block_index
                return remote_path
            else:
                # Get subcomponent from the block bridge
                subcomponent_name = parts[2]
                if (
                    hasattr(bridge_component, "_modules")
                    and subcomponent_name in bridge_component._modules
                ):
                    subcomponent_bridge = bridge_component._modules[subcomponent_name]
                    remote_path = f"{blocks_path}.{block_index}.{subcomponent_bridge.name}"
                    if last_component_only:
                        return subcomponent_bridge.name
                    return remote_path
                else:
                    raise ValueError(
                        f"Component {subcomponent_name} not found in blocks components"
                    )

        # For other nested paths, navigate through the bridge components
        remote_path = bridge_component.name
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        if last_component_only:
            return remote_path.split(".")[-1]
        return remote_path

    def convert_weights(self, hf_model: PreTrainedModel) -> dict[str, torch.Tensor]:
        """Convert the weights from the HuggingFace format to the HookedTransformer format.

        Args:
            hf_model: The HuggingFace model to convert.

        Returns:
            dict[str, torch.Tensor]: The converted weights.
        """
        if self.conversion_rules is None:
            raise ValueError("conversion_rules must be set before calling convert_weights")
        state_dict = self.conversion_rules.convert(input_value=hf_model)

        # Flatten state dictionary such that PyTorch can load it properly
        flattened_state_dict = self.flatten_nested_dict(state_dict)
        return flattened_state_dict

    def flatten_nested_dict(
        self,
        input: dict[str, torch.Tensor] | list[Any] | torch.Tensor,
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, torch.Tensor]:
        """
        Flattens a nested dictionary/list structure into a flat dictionary with dot notation.

        Args:
            input: The input structure (can be dict, list, or a value)
            parent_key: The parent key for the current item (used in recursion)
            sep: Separator to use between nested keys (default '.')

        Returns:
            dict: Flattened dictionary with dot notation keys
        """
        items = {}

        if isinstance(input, dict):
            for k, v in input.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v

        elif isinstance(input, list):
            for i, v in enumerate(input):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.update(self.flatten_nested_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        else:
            items[parent_key] = input

        return items
