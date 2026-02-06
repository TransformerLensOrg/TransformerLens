"""Utilities for traversing and applying functions to every component in a TransformerBridge model."""

from typing import Any, Callable

import torch.nn as nn

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def collect_all_submodules_of_component(
    model: TransformerBridge,
    component: GeneralizedComponent,
    submodules: dict,
    block_prefix: str = "",
) -> dict:
    """Recursively collects all submodules of a component in a TransformerBridge model.
    Args:
        model: The TransformerBridge model to collect submodules from
        component: The component to collect submodules from
        submodules: A dictionary to populate with submodules (modified in-place)
        block_prefix: Prefix for the block name, needed for components that are part of a block bridge
    Returns:
        Dictionary mapping submodule names to their respective submodules
    """
    for component_submodule in component.submodules.values():
        # Skip components without names (e.g., OPT's MLP container)
        if component_submodule.name is not None:
            submodules[block_prefix + component_submodule.name] = component_submodule

        # If the component is a list item, we need to collect all submodules of the block bridge
        if component_submodule.is_list_item:
            submodules = collect_components_of_block_bridge(model, component_submodule, submodules)

        # If the component has submodules, we need to collect them recursively
        if component_submodule.submodules:
            submodules = collect_all_submodules_of_component(
                model, component_submodule, submodules, block_prefix
            )
    return submodules


def collect_components_of_block_bridge(
    model: TransformerBridge, component: GeneralizedComponent, components: dict
) -> dict:
    """Collects all components of a BlockBridge component.
    Args:
        model: The TransformerBridge model to collect components from
        component: The BlockBridge component to collect components from
        components: A dictionary to populate with components (modified in-place)
    Returns:
        Dictionary mapping component names to their respective components
    """

    # Retrieve the remote component list from the adapter (we need a ModuleList to iterate over)
    if component.name is None:
        raise ValueError("Block bridge component must have a name")
    remote_module_list = model.adapter.get_remote_component(model.original_model, component.name)

    # Make sure the remote component is a ModuleList
    if isinstance(remote_module_list, nn.ModuleList):
        for block in remote_module_list:
            components[block.name] = block
            components = collect_all_submodules_of_component(model, block, components, block.name)
    return components


def collect_all_components(model: TransformerBridge, components: dict) -> dict:
    """Collects all components in a TransformerBridge inside a dictionary.
    The keys are the component names, and the values are the components themselves.
    Args:
        model: The TransformerBridge model to collect components from
        components: A dictionary to populate with components (modified in-place)
    Returns:
        Dictionary mapping component names to their respective components
    """

    # Iterate through all components in component mapping
    for component in model.adapter.get_component_mapping().values():
        components[component.name] = component
        components = collect_all_submodules_of_component(model, component, components)

        # We need to enable compatibility mode for all different blocks of the component if the component is a list item
        if component.is_list_item:
            components = collect_components_of_block_bridge(model, component, components)
    return components


def apply_fn_to_all_components(
    model: TransformerBridge,
    fn: Callable[[GeneralizedComponent], Any],
    components: dict | None = None,
) -> dict[str, Any]:
    """Applies a function to all components in the TransformerBridge model.
    Args:
        model: The TransformerBridge model to apply the function to
        fn: The function to apply to each component
        components: Optional dictionary of components to apply the function to, if None, all components are collected
    Returns:
        return_values: A dictionary mapping component names to the return values of the function
    """

    if components is None:
        components = collect_all_components(model, {})

    return_values = {}

    for component in components.values():
        return_values[component.name] = fn(component)

    return return_values
