from __future__ import annotations

"""Component creation utilities for creating bridged components."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.types import ComponentMapping, RemoteModel

if TYPE_CHECKING:
    pass


def create_bridged_component(
    remote_import: Tuple[str, type],
    remote_model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    name: str,
    config: Optional[Any] = None,
) -> nn.Module:
    """Create a bridged component from a RemoteImport.
    
    Args:
        remote_import: Tuple of (path, component_type)
        remote_model: The remote model to get the component from
        architecture_adapter: The architecture adapter
        name: The name of the component
        config: Optional configuration for the component
        
    Returns:
        Bridged component instance
    """
    if not isinstance(remote_import, tuple) or len(remote_import) != 2:
        raise ValueError("RemoteImport must be a tuple of (path, component_type)")

    path, component_type = remote_import
    original_component = architecture_adapter.get_remote_component(remote_model, path)
    
    # Create the bridge component with name and config
    bridge_component = component_type(name=name, config=config)
    
    # Set the original component
    bridge_component.set_original_component(original_component)
    
    return bridge_component


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


def create_and_replace_components_from_mapping(
    component_mapping: ComponentMapping,
    model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    bridge: Any = None,
) -> RemoteModel:
    """Create and replace components on a remote model from a mapping.
    
    Args:
        component_mapping: The component mapping to process
        model: The remote model to modify
        architecture_adapter: The architecture adapter
        bridge: Optional bridge to set attributes on
        
    Returns:
        The modified remote model
    """
    for tl_path, remote_spec in component_mapping.items():
        if isinstance(remote_spec, tuple) and len(remote_spec) == 3:
            # This is a BlockMapping with sub-components
            remote_path_template, bridge_type_or_config, sub_mapping = remote_spec
            original_block_container = architecture_adapter.get_remote_component(
                model, remote_path_template
            )
            
            # Extract component type and config
            if isinstance(bridge_type_or_config, tuple) and len(bridge_type_or_config) == 2:
                bridge_type, config = bridge_type_or_config
            else:
                bridge_type = bridge_type_or_config
                config = None
            
            if isinstance(original_block_container, nn.ModuleList):
                # Handle ModuleList of blocks
                bridged_block_list = nn.ModuleList()
                for i, original_block in enumerate(original_block_container):
                    # Create block bridge
                    bridged_block = bridge_type(name=f"{tl_path}.{i}", config=config)
                    bridged_block.set_original_component(original_block)
                    
                    # Recursively process sub-components
                    create_and_replace_components_from_mapping(
                        sub_mapping,
                        bridged_block.original_component,
                        architecture_adapter,
                    )
                    bridged_block_list.append(bridged_block)
                
                replace_remote_component(bridged_block_list, remote_path_template, model)
            else:
                # Handle single block
                bridged_block = bridge_type(name=tl_path, config=config)
                bridged_block.set_original_component(original_block_container)
                
                # Recursively process sub-components
                create_and_replace_components_from_mapping(
                    sub_mapping,
                    bridged_block.original_component,
                    architecture_adapter,
                )
                replace_remote_component(bridged_block, remote_path_template, model)
                
        elif isinstance(remote_spec, tuple) and len(remote_spec) == 2:
            # This is a RemoteImport (path, component_type) or (path, (component_type, config))
            path, component_spec = remote_spec
            
            # Extract component type and config
            if isinstance(component_spec, tuple) and len(component_spec) == 2:
                component_type, config = component_spec
            else:
                component_type = component_spec
                config = None
            
            bridged_component = create_bridged_component(
                (path, component_type),
                model,
                architecture_adapter,
                name=tl_path,
                config=config,
            )
            replace_remote_component(bridged_component, path, model)

    # Set bridge attributes if bridge is provided
    if bridge:
        for tl_path in component_mapping:
            remote_path = architecture_adapter.translate_transformer_lens_path(tl_path)
            bridged_component = architecture_adapter.get_remote_component(model, remote_path)
            setattr(bridge, tl_path, bridged_component)

    return model


def set_original_components_from_mapping(
    component_mapping: ComponentMapping,
    model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    bridge: Any,
) -> RemoteModel:
    """Set original components on pre-existing bridge components from a mapping.
    
    Args:
        component_mapping: The component mapping to process
        model: The remote model to get components from
        architecture_adapter: The architecture adapter
        bridge: The bridge with pre-created bridge components
        
    Returns:
        The modified remote model
    """
    for tl_path, remote_spec in component_mapping.items():
        if isinstance(remote_spec, tuple) and len(remote_spec) == 3:
            # This is a BlockMapping with sub-components
            remote_path_template, bridge_type_or_config, sub_mapping = remote_spec
            original_block_container = architecture_adapter.get_remote_component(
                model, remote_path_template
            )
            
            if isinstance(original_block_container, nn.ModuleList):
                # Handle ModuleList of blocks
                for i, original_block in enumerate(original_block_container):
                    # Get the pre-created bridge block
                    bridge_block = bridge.blocks[i]
                    bridge_block.set_original_component(original_block)
                    
                    # Recursively set original components for sub-components
                    set_original_components_from_mapping(
                        sub_mapping,
                        bridge_block.original_component,
                        architecture_adapter,
                        bridge_block,
                    )
                    
                    # Replace the original block with the bridge block
                    original_block_container[i] = bridge_block
            else:
                # Handle single block
                bridge_block = getattr(bridge, tl_path)
                bridge_block.set_original_component(original_block_container)
                
                # Recursively set original components for sub-components
                set_original_components_from_mapping(
                    sub_mapping,
                    bridge_block.original_component,
                    architecture_adapter,
                    bridge_block,
                )
                
                # Replace the original component with the bridge
                replace_remote_component(bridge_block, remote_path_template, model)
                
        elif isinstance(remote_spec, tuple) and len(remote_spec) == 2:
            # This is a RemoteImport (path, component_type) or (path, (component_type, config))
            path, component_spec = remote_spec
            
            # Get the original component
            original_component = architecture_adapter.get_remote_component(model, path)
            
            # Get the pre-created bridge component
            bridge_component = getattr(bridge, tl_path)
            bridge_component.set_original_component(original_component)
            
            # Replace the original component with the bridge
            replace_remote_component(bridge_component, path, model)

    return model
