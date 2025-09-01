"""Utilities for handling hook aliases in the bridge system."""

import warnings
from typing import Any, Dict, List, Optional, Set, Union


def resolve_alias(
    target_object: Any,
    requested_name: str,
    aliases: Dict[str, str] | Dict[str, Union[str, List[str]]],
) -> Optional[Any]:
    """Resolve a hook alias to the actual hook object.

    Args:
        target_object: The object to get the resolved attribute from
        requested_name: The name being requested (potentially an alias)
        aliases: Dictionary mapping alias names to target names

    Returns:
        The resolved hook object if alias found, None otherwise
    """
    if requested_name in aliases:
        target_name = aliases[requested_name]

        if hasattr(target_object, "disable_warnings") and target_object.disable_warnings == False:
            warnings.warn(
                f"Hook '{requested_name}' is deprecated and will be removed in a future version. "
                f"Use '{target_name}' instead.",
                FutureWarning,
                stacklevel=3,  # Adjusted for utility function call
            )

        def _resolve_single_target(target_name: str) -> Any:
            """Helper function to resolve a single target name."""
            target_name_split = target_name.split(".")
            # there are multiple target names, so we need to check all of them
            # this is the case for hook_pos_embed, which can be either pos_embed.hook_out (gpt2-style) or rotary_emb.hook_out (gemma/etc-style)
            if len(target_name_split) > 1:
                current_attr = target_object
                for i in range(len(target_name_split) - 1):
                    if not hasattr(current_attr, target_name_split[i]):
                        continue
                    current_attr = getattr(current_attr, target_name_split[i])

                # Check if the final attribute exists
                if not hasattr(current_attr, target_name_split[-1]):
                    raise AttributeError(
                        f"'{type(current_attr).__name__}' object has no attribute '{target_name_split[-1]}'"
                    )
                next_attr = getattr(current_attr, target_name_split[-1])
                return next_attr
            else:
                # Check if the target attribute exists before getting it
                if not hasattr(target_object, target_name):
                    raise AttributeError(
                        f"'{type(target_object).__name__}' object has no attribute '{target_name}'"
                    )
                # Return the target hook
                return getattr(target_object, target_name)

        # if the target_name is a list, we check all elements
        if isinstance(target_name, list):
            for target_name_item in target_name:
                try:
                    result = _resolve_single_target(target_name_item)
                    return result
                except AttributeError:
                    continue
            # If we get here, none of the targets in the list were found
            raise AttributeError(
                f"None of the target names {target_name} could be resolved on '{type(target_object).__name__}' object"
            )
        else:
            return _resolve_single_target(target_name)
    return None


def _collect_aliases_from_module(
    module: Any, path: str, aliases: Dict[str, str], visited: Set[int] = set()
) -> None:
    """Helper function to collect all aliases from a single module.
    Args:
        module: The module to collect aliases from
        path: Current path prefix for building full names
        aliases: Dictionary to populate with aliases (modified in-place)
        visited: Set of already visited module IDs to prevent infinite recursion
    """
    mod_id = id(module)
    if mod_id in visited:
        return
    visited.add(mod_id)

    if hasattr(module, "hook_aliases"):
        for alias_name, target_name in module.hook_aliases.items():
            if alias_name == "":
                # Empty string creates cache alias: embed -> embed.hook_out
                if path:  # Only add if we have a meaningful path
                    aliases[path] = f"{path}.{target_name}"
            else:
                # Named hook alias: embed.hook_embed -> embed.hook_out
                # Handle special case, hook_pos_embed and hook_embed should not be prefixed
                if path and not (alias_name == "hook_pos_embed" or alias_name == "hook_embed"):
                    full_alias = f"{path}.{alias_name}"
                    full_target = f"{path}.{target_name}"
                else:
                    full_alias = alias_name
                    full_target = f"{path}.{target_name}" if path else target_name

                aliases[full_alias] = full_target

    # Recursively collect from submodules, excluding original_model
    if hasattr(module, "named_children"):
        for child_name, child_module in module.named_children():
            # Skip the original_model to avoid collecting hooks from HuggingFace model
            if child_name == "original_model" or child_name == "_original_component":
                continue

            child_path = f"{path}.{child_name}" if path else child_name
            _collect_aliases_from_module(child_module, child_path, aliases, visited)


def collect_aliases_recursive(module: Any, prefix: str = "") -> Dict[str, str]:
    """Recursively collect all aliases from a module and its children.
    This unified function collects both:
    - Named hook aliases: old_hook_name -> new_hook_name
    - Cache aliases: component_name -> component_name.hook_out (from empty string keys)
    Args:
        module: The module to collect aliases from
        prefix: Path prefix for building full names
    Returns:
        Dictionary mapping all alias names to target names
    """
    aliases: Dict[str, str] = {}
    visited: Set[int] = set()
    _collect_aliases_from_module(module, prefix, aliases, visited)
    return aliases
