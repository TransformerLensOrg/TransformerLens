from __future__ import annotations

"""Component creation utilities for creating bridged components."""

from typing import TYPE_CHECKING, Any, Optional

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.types import RemoteModel

if TYPE_CHECKING:
    pass


def create_bridged_component(
    remote_model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    name: str,
    component_type: type[GeneralizedComponent],
    config: Optional[Any] = None,
) -> GeneralizedComponent:
    """Create a bridged component from a remote model.
    
    Args:
        remote_model: The remote model to get the component from
        architecture_adapter: The architecture adapter
        name: The name/path of the component in the remote model
        component_type: The bridge component class to create
        config: Optional configuration for the component
        
    Returns:
        Bridged component instance
    """
    # Get the original component from the remote model
    original_component = architecture_adapter.get_remote_component(remote_model, name)
    
    # Create the bridge component with name and config
    bridge_component = component_type(name=name, config=config)
    
    # Set the original component
    bridge_component.set_original_component(original_component)
    
    return bridge_component


def replace_remote_component(
    replacement_component: GeneralizedComponent,
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
