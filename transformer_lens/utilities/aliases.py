"""Utilities for handling hook aliases in the bridge system."""

import warnings
from typing import Any, Dict, Optional, Set


def resolve_hook_alias(
    target_object: Any, requested_name: str, hook_aliases: Dict[str, str]
) -> Optional[Any]:
    """Resolve a hook alias to the actual hook object.

    Args:
        target_object: The object to get the resolved attribute from
        requested_name: The name being requested (potentially an alias)
        hook_aliases: Dictionary mapping alias names to target names

    Returns:
        The resolved hook object if alias found, None otherwise
    """
    if requested_name in hook_aliases:
        target_hook = hook_aliases[requested_name]
        warnings.warn(
            f"Hook '{requested_name}' is deprecated and will be removed in a future version. "
            f"Use '{target_hook}' instead.",
            FutureWarning,
            stacklevel=3,  # Adjusted for utility function call
        )
        # Return the target hook
        return getattr(target_object, target_hook)
    return None


def _collect_aliases_from_module(
    module: Any, path: str, aliases: Dict[str, str], visited: Set[int]
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
                if path:
                    full_alias = f"{path}.{alias_name}"
                    full_target = f"{path}.{target_name}"
                else:
                    full_alias = alias_name
                    full_target = target_name
                aliases[full_alias] = full_target

    # Recursively collect from submodules, excluding original_model
    if hasattr(module, "named_children"):
        for child_name, child_module in module.named_children():
            # Skip the original_model to avoid collecting hooks from HuggingFace model
            if child_name == "original_model":
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
