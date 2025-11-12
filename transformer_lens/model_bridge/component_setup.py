from __future__ import annotations

"""Component setup utilities for creating and configuring bridged components."""

import copy
from typing import Any, cast

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.types import RemoteModel


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
    setup_components(component_mapping, bridge_module, architecture_adapter, original_model)


def setup_submodules(
    component: GeneralizedComponent,
    architecture_adapter: ArchitectureAdapter,
    original_model: RemoteModel,
) -> None:
    """Set up submodules for a bridge component using proper component setup.

    Args:
        component: The bridge component to set up submodules for
        architecture_adapter: The architecture adapter
        original_model: The original model to get components from
    """
    for module_name, submodule in component.submodules.items():
        if submodule.is_list_item:
            # Submodule is a BlockBridge - create a ModuleList of bridge components
            if submodule.name is None:
                raise ValueError(f"List item component {module_name} must have a name")
            bridged_list = setup_blocks_bridge(submodule, architecture_adapter, original_model)
            # Set the list on the bridge module as a proper module
            component.add_module(module_name, bridged_list)

            replace_remote_component(bridged_list, submodule.name, original_model)
        # Only add if not already registered as a PyTorch module
        if module_name not in component._modules:
            # Get original component (use parent if no container, e.g. OPT's MLP)
            if submodule.name is None:
                original_subcomponent = original_model
            else:
                remote_path = submodule.name
                original_subcomponent = architecture_adapter.get_remote_component(
                    original_model, remote_path
                )

            submodule.set_original_component(original_subcomponent)
            setup_submodules(submodule, architecture_adapter, original_subcomponent)
            component.add_module(module_name, submodule)

            # Replace original with bridge (skip if no container)
            if submodule.name is not None:
                replace_remote_component(submodule, submodule.name, original_model)

    # Note: Alias registration happens later in enable_compatibility_mode()
    # after weight processing to ensure aliases point to processed weights


def setup_components(
    components: dict[str, Any],
    bridge_module: nn.Module,
    architecture_adapter: ArchitectureAdapter,
    original_model: RemoteModel,
) -> None:
    """Set up components on the bridge module.

    Args:
        components: Dictionary of component name to bridge component mappings
        bridge_module: The bridge module to configure
        architecture_adapter: The architecture adapter
        original_model: The original model to get components from
    """
    for tl_path, bridge_component in components.items():
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

            # Set up submodules for this component
            setup_submodules(bridge_component, architecture_adapter, original_component)

            # Set the bridge component on the bridge module as a proper module
            bridge_module.add_module(tl_path, bridge_component)

            # Replace the original component with the bridge component
            replace_remote_component(bridge_component, remote_path, original_model)


def setup_blocks_bridge(
    blocks_template: Any, architecture_adapter: ArchitectureAdapter, original_model: RemoteModel
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

    # Ensure the original blocks container is iterable
    if not hasattr(original_blocks, "__iter__"):
        raise TypeError(f"Component {blocks_template.name} is not iterable")

    # Create a new ModuleList of bridge components
    bridged_blocks = nn.ModuleList()

    # Cast to indicate to mypy that original_blocks is iterable after the check
    iterable_blocks = cast(Any, original_blocks)
    for i, original_block in enumerate(iterable_blocks):
        # Create a copy of the template bridge for this block
        block_bridge = copy.deepcopy(blocks_template)
        block_bridge.name = f"{blocks_template.name}.{i}"

        # Set the original component for this block
        block_bridge.set_original_component(original_block)

        # Set up submodules for this block component
        setup_submodules(block_bridge, architecture_adapter, original_block)

        bridged_blocks.append(block_bridge)

    # Replace the original blocks with the bridged blocks
    replace_remote_component(bridged_blocks, blocks_template.name, original_model)

    return bridged_blocks
