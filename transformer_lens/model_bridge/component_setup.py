from __future__ import annotations

"Component setup utilities for creating and configuring bridged components."
import copy
import logging
from typing import TYPE_CHECKING, Any, cast

logger = logging.getLogger(__name__)

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.symbolic import SymbolicBridge
from transformer_lens.model_bridge.types import RemoteModel

if TYPE_CHECKING:
    pass


def replace_remote_component(
    replacement_component: nn.Module, remote_path: str, remote_model: RemoteModel
) -> None:
    """Replace a component in a remote model.

    Args:
        replacement_component: The new component to install
        remote_path: Path to the component in the remote model
        remote_model: The remote model to modify
    """
    path_parts = remote_path.split(".")
    current = remote_model
    for part in path_parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            raise ValueError(f"Path {remote_path} not found in model")
    target_attr = path_parts[-1]
    if hasattr(current, target_attr):
        setattr(current, target_attr, replacement_component)
    else:
        raise ValueError(f"Attribute {target_attr} not found in {current}")


def set_original_components(
    bridge_module: nn.Module, architecture_adapter: ArchitectureAdapter, original_model: RemoteModel
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
    skipped_optional: list[str] = []
    for module_name, submodule in component.submodules.items():
        if submodule.is_list_item:
            if submodule.name is None:
                raise ValueError(f"List item component {module_name} must have a name")
            bridged_list = setup_blocks_bridge(submodule, architecture_adapter, original_model)
            component.add_module(module_name, bridged_list)
            replace_remote_component(bridged_list, submodule.name, original_model)
            # Add to real_components mapping
            component.real_components[module_name] = (submodule.name, list(bridged_list))
        elif isinstance(submodule, SymbolicBridge):
            # SymbolicBridge: no real component; set up submodules via parent's model
            setup_submodules(submodule, architecture_adapter, original_model)

            # Add the symbolic bridge as a module (for structural access like blocks[i].mlp.in)
            if module_name not in component._modules:
                component.add_module(module_name, submodule)

            # Add symbolic bridge's real_components to parent's mapping with prefixed keys
            for sub_name, (sub_path, sub_comp) in submodule.real_components.items():
                prefixed_key = f"{module_name}.{sub_name}"
                component.real_components[prefixed_key] = (sub_path, sub_comp)
        else:
            # Set up original_component if not already set
            if submodule.original_component is None:
                if submodule.name is None:
                    original_subcomponent = original_model
                else:
                    remote_path = submodule.name
                    is_optional = getattr(submodule, "optional", False)
                    # Fast path: first segment absent or None → skip
                    first_segment = remote_path.split(".")[0]
                    first_value = getattr(original_model, first_segment, None)
                    if is_optional and first_value is None:
                        logger.debug(
                            "Optional '%s' (path '%s') absent on %s",
                            module_name,
                            remote_path,
                            getattr(component, "name", "?"),
                        )
                        skipped_optional.append(module_name)
                        continue
                    # Full resolution — catches deeper path failures (e.g. stub self_attn missing q_proj)
                    try:
                        original_subcomponent = architecture_adapter.get_remote_component(
                            original_model, remote_path
                        )
                    except AttributeError:
                        if is_optional:
                            logger.debug(
                                "Optional '%s' (path '%s') partially absent on %s",
                                module_name,
                                remote_path,
                                getattr(component, "name", "?"),
                            )
                            skipped_optional.append(module_name)
                            continue
                        raise
                submodule.set_original_component(original_subcomponent)
                setup_submodules(submodule, architecture_adapter, original_subcomponent)
                if submodule.name is not None:
                    replace_remote_component(submodule, submodule.name, original_model)

            # Add to _modules if not already present
            if module_name not in component._modules:
                component.add_module(module_name, submodule)

            # Add to real_components mapping (for non-list components)
            if not submodule.is_list_item and submodule.name is not None:
                component.real_components[module_name] = (submodule.name, submodule)

    # Clean up so architecture_adapter traversal won't find stale entries
    for name in skipped_optional:
        component.submodules.pop(name, None)


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
            bridged_list = setup_blocks_bridge(
                bridge_component, architecture_adapter, original_model
            )
            bridge_module.add_module(tl_path, bridged_list)
            replace_remote_component(bridged_list, remote_path, original_model)
            # Add to bridge module's real_components if it has the attribute
            if hasattr(bridge_module, "real_components"):
                bridge_module.real_components[tl_path] = (remote_path, list(bridged_list))  # type: ignore[index, assignment, operator]
        else:
            original_component = architecture_adapter.get_remote_component(
                original_model, remote_path
            )
            bridge_component.set_original_component(original_component)
            setup_submodules(bridge_component, architecture_adapter, original_component)
            bridge_module.add_module(tl_path, bridge_component)
            replace_remote_component(bridge_component, remote_path, original_model)
            # Add to bridge module's real_components if it has the attribute
            if hasattr(bridge_module, "real_components"):
                bridge_module.real_components[tl_path] = (remote_path, bridge_component)  # type: ignore[index, assignment, operator]


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
    original_blocks = architecture_adapter.get_remote_component(
        original_model, blocks_template.name
    )
    if not hasattr(original_blocks, "__iter__"):
        raise TypeError(f"Component {blocks_template.name} is not iterable")
    bridged_blocks = nn.ModuleList()
    iterable_blocks = cast(Any, original_blocks)
    for i, original_block in enumerate(iterable_blocks):
        block_bridge = copy.deepcopy(blocks_template)
        block_bridge.name = f"{blocks_template.name}.{i}"
        block_bridge.set_original_component(original_block)
        setup_submodules(block_bridge, architecture_adapter, original_block)
        bridged_blocks.append(block_bridge)
    replace_remote_component(bridged_blocks, blocks_template.name, original_model)
    return bridged_blocks
