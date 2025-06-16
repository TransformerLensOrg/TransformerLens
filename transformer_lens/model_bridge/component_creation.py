from __future__ import annotations

"""Component creation utilities for creating bridged components."""

from typing import TYPE_CHECKING, Any

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.types import ComponentMapping, RemoteModel

if TYPE_CHECKING:
    pass


def create_bridged_component(
    remote_import,
    remote_model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    name: str,
) -> nn.Module:
    """Create a bridged component from a RemoteImport."""
    if not isinstance(remote_import, tuple) or len(remote_import) != 2:
        raise ValueError("RemoteImport must be a tuple of (path, component_type)")

    path, component_type = remote_import
    original_component = architecture_adapter.get_remote_component(remote_model, path)
    return component_type(
        original_component=original_component,
        name=name,
        architecture_adapter=architecture_adapter,
    )


def replace_remote_component(
    bridged_component: nn.Module, remote_path: str, remote_model: nn.Module
) -> None:
    """Replace a component on the remote model."""
    path_parts = remote_path.split(".")
    parent_obj = remote_model
    for part in path_parts[:-1]:
        if part.isdigit():
            parent_obj = parent_obj[int(part)]  # type: ignore[index]
        else:
            parent_obj = getattr(parent_obj, part)
    setattr(parent_obj, path_parts[-1], bridged_component)


def create_and_replace_components_from_mapping(
    component_mapping: ComponentMapping,
    model: RemoteModel,
    architecture_adapter: "ArchitectureAdapter",
    bridge: Any = None,
) -> RemoteModel:
    """Create and replace components on a remote model from a mapping."""
    for tl_path, remote_spec in component_mapping.items():
        if isinstance(remote_spec, tuple) and len(remote_spec) == 3:
            # This is a BlockMapping
            remote_path_template, bridge_type, sub_mapping = remote_spec
            original_block_container = architecture_adapter.get_remote_component(
                model, remote_path_template
            )
            if isinstance(original_block_container, nn.ModuleList):
                bridged_block_list = nn.ModuleList()
                for i, original_block in enumerate(original_block_container):
                    bridged_block = bridge_type(
                        original_component=original_block,
                        name=f"{tl_path}.{i}",
                        architecture_adapter=architecture_adapter,
                    )
                    create_and_replace_components_from_mapping(
                        sub_mapping,
                        bridged_block.original_component,
                        architecture_adapter,
                    )
                    bridged_block_list.append(bridged_block)
                replace_remote_component(bridged_block_list, remote_path_template, model)
            else:
                bridged_block = bridge_type(
                    original_component=original_block_container,
                    name=tl_path,
                    architecture_adapter=architecture_adapter,
                )
                create_and_replace_components_from_mapping(
                    sub_mapping,
                    bridged_block.original_component,
                    architecture_adapter,
                )
                replace_remote_component(bridged_block, remote_path_template, model)
        elif isinstance(remote_spec, tuple) and len(remote_spec) == 2:
            # This is a RemoteImport
            bridged_component = create_bridged_component(
                remote_spec,
                model,
                architecture_adapter,
                name=tl_path,
            )
            replace_remote_component(bridged_component, remote_spec[0], model)

    if bridge:
        for tl_path in component_mapping:
            remote_path = architecture_adapter.translate_transformer_lens_path(tl_path)
            bridged_component = architecture_adapter.get_remote_component(model, remote_path)
            setattr(bridge, tl_path, bridged_component)

    return model
