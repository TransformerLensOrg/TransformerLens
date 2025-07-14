from __future__ import annotations

"""Component setup utilities for creating and configuring bridged components."""

import copy
from typing import TYPE_CHECKING, Any, Optional

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.types import RemoteModel

if TYPE_CHECKING:
    pass


def replace_remote_component(
    replacement_component: nn.Module,
    remote_path: str,
    remote_model: RemoteModel,
) -> None:
    """Replace a component in a remote model.

    Args:
        replacement_component: The new component to install
        remote_path: Path to the component in the remote model
        remote_model: The remote model to modify
    """
    # Split the path into parts
    path_parts = remote_path.split(".")

    # Navigate to the parent of the target component
    current = remote_model
    for part in path_parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            raise ValueError(f"Path {remote_path} not found in model")

    # Replace the target component
    target_attr = path_parts[-1]
    if hasattr(current, target_attr):
        setattr(current, target_attr, replacement_component)
    else:
        raise ValueError(f"Attribute {target_attr} not found in {current}")


def set_original_components(
    bridge_module: nn.Module,
    architecture_adapter: ArchitectureAdapter,
    original_model: RemoteModel,
) -> None:
    """Set original components on the pre-created bridge components.
    
    Args:
        bridge_module: The bridge module to configure
        architecture_adapter: The architecture adapter
        original_model: The original model to get components from
    """
    component_mapping = architecture_adapter.get_component_mapping()

    # Modern bridge instance mapping - set original components directly
    for tl_path, bridge_component in component_mapping.items():
        remote_path = bridge_component.name
        if bridge_component.is_list_item:
            # Special handling for list items - create a ModuleList of bridge components
            bridged_list = setup_blocks_bridge(
                bridge_component, architecture_adapter, original_model
            )
            # Set the list on the bridge module as a proper module
            bridge_module.add_module(tl_path, bridged_list)
            replace_remote_component(bridged_list, remote_path, original_model)
        else:
            # Regular component handling
            original_component = architecture_adapter.get_remote_component(
                original_model, remote_path
            )
            bridge_component.set_original_component(original_component)

            # Set the bridge component on the bridge module as a proper module
            bridge_module.add_module(tl_path, bridge_component)

            # Replace the original component with the bridge component
            replace_remote_component(bridge_component, remote_path, original_model)


def setup_blocks_bridge(
    blocks_template: Any, 
    architecture_adapter: ArchitectureAdapter, 
    original_model: RemoteModel
) -> nn.ModuleList:
    """Set up blocks bridge with proper ModuleList structure.
    
    Args:
        blocks_template: Template bridge component for blocks
        architecture_adapter: The architecture adapter
        original_model: The original model to get components from
        
    Returns:
        ModuleList of bridged block components
    """
    # Get the original blocks container
    original_blocks = architecture_adapter.get_remote_component(
        original_model, blocks_template.name
    )

    # Create a new ModuleList of bridge components
    bridged_blocks = nn.ModuleList()

    for i, original_block in enumerate(original_blocks):
        # Create a copy of the template bridge for this block
        block_bridge = copy.deepcopy(blocks_template)
        block_bridge.name = f"{blocks_template.name}.{i}"

        # Set the original component for this block
        block_bridge.set_original_component(original_block)

        # Set original components for all submodules
        if hasattr(block_bridge, "_modules"):
            for submodule_name, submodule in block_bridge._modules.items():
                if (
                    hasattr(submodule, "set_original_component")
                    and submodule_name != "hook_in"
                    and submodule_name != "hook_out"
                ):
                    # Get the original subcomponent
                    original_subcomponent = getattr(original_block, submodule.name)
                    submodule.set_original_component(original_subcomponent)

                    # Handle nested submodules (like attention projections)
                    if hasattr(submodule, "_modules"):
                        for nested_name, nested_module in submodule._modules.items():
                            if (
                                hasattr(nested_module, "set_original_component")
                                and nested_name != "hook_in"
                                and nested_name != "hook_out"
                            ):
                                original_nested = getattr(
                                    original_subcomponent, nested_module.name
                                )
                                nested_module.set_original_component(original_nested)

        bridged_blocks.append(block_bridge)

    # Replace the original blocks with the bridged blocks
    replace_remote_component(bridged_blocks, blocks_template.name, original_model)

    return bridged_blocks
