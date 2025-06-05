from __future__ import annotations

"""Component creation utilities for creating bridged components."""

from typing import Any, Type

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.types import RemoteImport, RemoteModel


def create_bridged_component(
    remote_import,
    remote_model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    prepend: str | None = None,
) -> nn.Module:
    """Create a bridged component from a RemoteImport.

    This function takes a RemoteImport (path and component type), a remote model,
    and an architecture adapter, then creates and returns the appropriate bridged component.

    Args:
        remote_import: A tuple containing (path, component_type)
        remote_model: The remote model to extract the component from
        architecture_adapter: The architecture adapter to use for component creation
        prepend: Optional path to prepend to the component path (e.g. "blocks.0")

    Returns:
        The created bridged component

    Raises:
        ValueError: If the remote import structure is invalid
        AttributeError: If the component cannot be found in the remote module
    """
    if not isinstance(architecture_adapter, ArchitectureAdapter):
        raise TypeError("architecture_adapter must be an instance of ArchitectureAdapter")

    if not isinstance(remote_import, tuple) or len(remote_import) != 2:
        raise ValueError("RemoteImport must be a tuple of (path, component_type)")

    path, component_type = remote_import

    # If prepend is set, modify the path
    full_path = f"{prepend}.{path}" if prepend else path

    # Get the original component from the remote model
    original_component = architecture_adapter.get_remote_component(remote_model, full_path)

    # Create and return the bridged component
    return component_type(
        original_component=original_component,
        name=full_path,
        architecture_adapter=architecture_adapter
    ) 

def replace_remote_component(
    bridged_component: nn.Module, remote_path: str, remote_model: nn.Module
) -> None:
    """Replace a component on the remote model.

    This works by grabbing everything in the path before the last part, and then
    setting the property in that object to the bridged component based on the
    last part of the path.

    Args:
        bridged_component: The new, bridged component.
        remote_path: The full path to the component.
        remote_model: The remote model to modify.
    """
    path_parts = remote_path.split(".")
    parent_obj = remote_model
    for part in path_parts[:-1]:
        if part.isdigit():
            parent_obj = parent_obj[int(part)]
        else:
            parent_obj = getattr(parent_obj, part)
    setattr(parent_obj, path_parts[-1], bridged_component) 