"""Architecture adapter base class.

This module contains the base class for architecture adapters that map between different model architectures.
"""

from typing import Any, cast

import torch
from torch import nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import HookConversionSet
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
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

    default_cfg: dict[str, Any] = {}

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        """Initialize the architecture adapter.

        Args:
            cfg: The configuration object.
        """
        self.cfg = cfg

        self.component_mapping: ComponentMapping | None = None
        self.conversion_rules: HookConversionSet | None = None

        # Merge default_cfg into cfg for missing variables
        self._merge_default_config()

    def _merge_default_config(self) -> None:
        """Merge default_cfg into cfg for variables that don't exist in cfg."""
        for key, value in self.default_cfg.items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

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

    def get_component_from_list_module(
        self,
        list_module: RemoteComponent,
        bridge_component: GeneralizedComponent,
        parts: list[str],
    ) -> RemoteComponent:
        """Get a component from a list module using the bridge component and the transformer lens path.
        Args:
            list_module: The remote list module to get the component from
            bridge_component: The bridge component
            parts: The parts of the transformer lens path to navigate
        Returns:
            The requested component from the list module described by the path
        """

        # Handle list item indexing (like blocks)
        item_index = parts[1]
        if not item_index.isdigit():
            raise ValueError(f"Expected item index, got {item_index}")

        if not hasattr(list_module, "__getitem__"):
            raise TypeError(f"Component {bridge_component.name} is not indexable")

        # Cast to indicate to mypy that list_module is indexable after the check
        indexable_container = cast(Any, list_module)
        item = indexable_container[int(item_index)]

        if len(parts) == 2:
            # Just return the item
            return item
        else:
            # Get subcomponent from the item using bridge mapping
            subcomponent_name = parts[2]

            # Check the submodules attribute for bridge submodules
            if subcomponent_name in bridge_component.submodules:
                subcomponent_bridge = bridge_component.submodules[subcomponent_name]

                # If there are more parts (like blocks.0.attn.W_Q), navigate deeper
                if len(parts) > 3:
                    # Navigate through the deeper subcomponents
                    current_bridge = subcomponent_bridge
                    current = getattr(item, subcomponent_bridge.name)

                    for i in range(3, len(parts)):
                        deeper_component_name = parts[i]

                        if deeper_component_name.isdigit() and current_bridge.is_list_item:
                            # We are dealing with a nested BlockBridge, call the function recursively
                            # and pass the path (parts) starting from the nested BlockBridge
                            return self.get_component_from_list_module(
                                current, current_bridge, parts[i - 1 :]
                            )

                        # Check submodules for deeper components
                        if deeper_component_name in current_bridge.submodules:
                            current_bridge = current_bridge.submodules[deeper_component_name]
                            current = getattr(current, current_bridge.name)
                        else:
                            raise ValueError(
                                f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                            )

                    return current
                else:
                    # Just the 3-level path
                    return getattr(item, subcomponent_bridge.name)
            else:
                raise ValueError(
                    f"Component {subcomponent_name} not found in {parts[0]} components"
                )

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
        if self.component_mapping is None or parts[0] not in self.component_mapping:
            raise ValueError(f"Component {parts[0]} not found in component mapping")

        bridge_component = self.component_mapping[parts[0]]

        if len(parts) == 1:
            # Simple case: just return the component at the bridge's remote path
            return self.get_remote_component(model, bridge_component.name)

        # For nested paths like "blocks.0.attn", we need to handle the indexing
        if bridge_component.is_list_item and len(parts) >= 2:
            # Get the remote ModuleList for the indexed item
            list_module = self.get_remote_component(model, bridge_component.name)
            return self.get_component_from_list_module(list_module, bridge_component, parts)

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
        if bridge_component.is_list_item and len(parts) >= 2:
            # Handle list item indexing (like blocks)
            item_index = parts[1]
            if not item_index.isdigit():
                raise ValueError(f"Expected item index, got {item_index}")

            # Get the base items path
            items_path = bridge_component.name

            if len(parts) == 2:
                # Just return the indexed item path
                remote_path = f"{items_path}.{item_index}"
                if last_component_only:
                    return item_index
                return remote_path
            else:
                # Get subcomponent from the item bridge
                subcomponent_name = parts[2]

                # Check the submodules attribute for bridge submodules
                if subcomponent_name in bridge_component.submodules:
                    subcomponent_bridge = bridge_component.submodules[subcomponent_name]

                    # If there are more parts (like blocks.0.attn.q_proj), navigate deeper
                    if len(parts) > 3:
                        # Navigate through the deeper subcomponents
                        current_bridge = subcomponent_bridge
                        remote_path_parts = [items_path, item_index, subcomponent_bridge.name]

                        for i in range(3, len(parts)):
                            deeper_component_name = parts[i]

                            # Check submodules for deeper components
                            if deeper_component_name in current_bridge.submodules:
                                current_bridge = current_bridge.submodules[deeper_component_name]
                                remote_path_parts.append(current_bridge.name)
                            else:
                                raise ValueError(
                                    f"Component {deeper_component_name} not found in {'.'.join(parts[:i])} components"
                                )

                        remote_path = ".".join(remote_path_parts)
                        if last_component_only:
                            return current_bridge.name
                        return remote_path
                    else:
                        # Just the 3-level path
                        remote_path = f"{items_path}.{item_index}.{subcomponent_bridge.name}"
                        if last_component_only:
                            return subcomponent_bridge.name
                        return remote_path
                else:
                    raise ValueError(
                        f"Component {subcomponent_name} not found in {parts[0]} components"
                    )

        # For other nested paths, navigate through the bridge components
        remote_path = bridge_component.name
        if len(parts) > 1:
            remote_path = f"{remote_path}.{'.'.join(parts[1:])}"

        if last_component_only:
            return remote_path.split(".")[-1]
        return remote_path

    def convert_weights(self, hf_model: nn.Module) -> dict[str, torch.Tensor]:
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
