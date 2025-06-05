from __future__ import annotations

"""Component creation utilities for creating bridged components."""

from typing import Any, Type

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.types import (
    ComponentMapping,
    RemoteImport,
    RemoteModel,
)


def create_bridged_component(
    remote_import,
    remote_model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    name: str,
    prepend: str | None = None,
) -> nn.Module:
    """Create a bridged component from a RemoteImport.

    This function takes a RemoteImport (path and component type), a remote model,
    and an architecture adapter, then creates and returns the appropriate bridged component.

    Args:
        remote_import: A tuple containing (path, component_type)
        remote_model: The remote model to extract the component from
        architecture_adapter: The architecture adapter to use for component creation
        name: The name of the component in the TransformerLens model
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
        name=name,
        architecture_adapter=architecture_adapter,
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

def create_and_replace_components_from_mapping(
    component_mapping: ComponentMapping,
    remote_model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    remote_path_prepend: str | None = None,
    tl_path_prepend: str | None = None,
) -> None:
    """Create and replace components on a remote model from a mapping.

    This function iterates through a component mapping, creates bridged
    components, and replaces them on the remote model. It handles nested
    mappings and ModuleLists generically.

    Args:
        component_mapping: A dictionary mapping TransformerLens paths to remote
            components.
        remote_model: The remote model to modify.
        architecture_adapter: The architecture adapter to use for component
            creation.
        remote_path_prepend: A path to prepend to all remote component paths,
            used for recursion.
        tl_path_prepend: A path to prepend to all TransformerLens component paths,
            used for recursion.
    """
    for tl_path, remote_spec in component_mapping.items():
        full_tl_path = f"{tl_path_prepend}.{tl_path}" if tl_path_prepend else tl_path

        if isinstance(remote_spec, tuple) and len(remote_spec) == 3:
            # This is a BlockMapping (remote_path, bridge_type, sub_mapping)
            remote_path_template, bridge_type, sub_mapping = remote_spec

            full_remote_path_template = (
                f"{remote_path_prepend}.{remote_path_template}"
                if remote_path_prepend
                else remote_path_template
            )

            # Get the component this mapping refers to
            component_to_check = architecture_adapter.get_remote_component(
                remote_model, full_remote_path_template
            )

            if isinstance(component_to_check, nn.ModuleList):
                # It's a list of blocks, so we iterate
                for i in range(len(component_to_check)):
                    tl_item_path = f"{full_tl_path}.{i}"
                    remote_item_path = f"{full_remote_path_template}.{i}"

                    # Recurse for the sub-components within the item
                    create_and_replace_components_from_mapping(
                        sub_mapping,
                        remote_model,
                        architecture_adapter,
                        remote_path_prepend=remote_item_path,
                        tl_path_prepend=tl_item_path,
                    )

                    # Now bridge the block container itself
                    bridged_item = create_bridged_component(
                        (remote_item_path, bridge_type),
                        remote_model,
                        architecture_adapter,
                        name=tl_item_path,
                    )
                    replace_remote_component(
                        bridged_item, remote_item_path, remote_model
                    )
            else:
                # It's a single, non-list block-like component. Recurse.
                create_and_replace_components_from_mapping(
                    sub_mapping,
                    remote_model,
                    architecture_adapter,
                    remote_path_prepend=full_remote_path_template,
                    tl_path_prepend=full_tl_path,
                )
                # Bridge the container
                bridged_container = create_bridged_component(
                    (full_remote_path_template, bridge_type),
                    remote_model,
                    architecture_adapter,
                    name=full_tl_path,
                )
                replace_remote_component(
                    bridged_container, full_remote_path_template, remote_model
                )

        elif isinstance(remote_spec, tuple) and len(remote_spec) == 2:
            # This is a RemoteImport
            remote_path, _ = remote_spec
            full_remote_path = (
                f"{remote_path_prepend}.{remote_path}"
                if remote_path_prepend
                else remote_path
            )
            bridged_component = create_bridged_component(
                remote_spec,
                remote_model,
                architecture_adapter,
                name=full_tl_path,
                prepend=remote_path_prepend,
            )
            replace_remote_component(bridged_component, full_remote_path, remote_model) 