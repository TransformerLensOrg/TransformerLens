"""Base class for generalized transformer components."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities.aliases import resolve_alias


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.

    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    # Class attribute indicating whether this component represents a list item (like blocks)
    is_list_item: bool = False

    # Compatibility mode that can be activated/deactivated for legacy components/hooks
    compatibility_mode: bool = False
    # Whether to disable warnings about deprecated hooks
    disable_warnings: bool = False

    # Dictionary mapping deprecated hook names to their new equivalents
    # Subclasses can override this to define their own aliases
    hook_aliases: Dict[str, str] = {}
    property_aliases: Dict[str, str] = {}

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the generalized component.

        Args:
            name: The name of this component
            config: Optional configuration object for the component
            submodules: Dictionary of GeneralizedComponent submodules to register
            conversion_rule: Optional conversion rule for this component's hooks
        """
        super().__init__()
        self.name = name
        self.config = config
        self.submodules = submodules or {}
        self.conversion_rule = conversion_rule
        self._hook_registry: Dict[
            str, HookPoint
        ] = {}  # Dynamic registry of hook names to HookPoints

        # Standardized hooks for all bridge components
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

        # Apply conversion rule to hooks if available
        if self.conversion_rule is not None:
            self.hook_in.hook_conversion = self.conversion_rule
            self.hook_out.hook_conversion = self.conversion_rule

    def _register_hook(self, name: str, hook: HookPoint) -> None:
        """Register a hook in the component's hook registry."""
        # Set the name on the HookPoint
        hook.name = name
        # Add to registry
        self._hook_registry[name] = hook

    def get_hooks(self) -> Dict[str, HookPoint]:
        """Get all hooks registered in this component."""

        # Add aliases if compatibility mode is enabled
        if self.compatibility_mode and self.hook_aliases:
            # Only copy hook registry if compatibility mode is enabled to save memory
            hooks = self._hook_registry.copy()

            for alias_name, target_name in self.hook_aliases.items():
                # Use the existing alias system to resolve the target hook
                target_hook = resolve_alias(self, alias_name, self.hook_aliases)
                if target_hook is not None:
                    hooks[alias_name] = target_hook
            return hooks
        else:
            return self._hook_registry

    def _is_getattr_called_internally(self) -> bool:
        """This function checks if the __getattr__ method was being called internally
        (e.g by the setup process or run_with_cache).
        """
        # Look through the call stack
        for frame_info in inspect.stack():
            if "setup_components" in frame_info.function or "run_with_cache" in frame_info.function:
                return True
        return False

    def set_original_component(self, original_component: nn.Module) -> None:
        """Set the original component that this bridge wraps.

        Args:
            original_component: The original transformer component to wrap
        """
        self.add_module("_original_component", original_component)

    @property
    def original_component(self) -> Optional[nn.Module]:
        """Get the original component."""
        return self._modules.get("_original_component", None)

    def add_hook(self, hook_fn: Callable[..., torch.Tensor], hook_name: str = "output") -> None:
        """Add a hook function (HookedTransformer-compatible interface).

        Args:
            hook_fn: Function to call at this hook point
            hook_name: Name of the hook point (defaults to "output")
        """
        if hook_name == "output":
            self.hook_out.add_hook(hook_fn)
        elif hook_name == "input":
            self.hook_in.add_hook(hook_fn)
        else:
            raise ValueError(
                f"Hook name '{hook_name}' not supported. Supported names are 'output' and 'input'."
            )

    def remove_hooks(self, hook_name: str | None = None) -> None:
        """Remove hooks (HookedTransformer-compatible interface).

        Args:
            hook_name: Name of the hook point to remove. If None, removes all hooks.
        """
        if hook_name is None:
            self.hook_in.remove_hooks()
            self.hook_out.remove_hooks()
        elif hook_name == "output":
            self.hook_out.remove_hooks()
        elif hook_name == "input":
            self.hook_in.remove_hooks()
        else:
            raise ValueError(
                f"Hook name '{hook_name}' not supported. Supported names are 'output' and 'input'."
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Generic forward pass for bridge components with input/output hooks."""
        # Since we use add_module, the component is stored in _modules
        original_component = self._modules.get("_original_component", None)
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Try to find the main input
        input_arg_names = [
            "input",
            "hidden_states",
            "input_ids",
            "query_input",
            "x",
            "inputs_embeds",
        ]
        input_found = False
        # Try kwargs first
        for name in input_arg_names:
            if name in kwargs:
                kwargs[name] = self.hook_in(kwargs[name])
                input_found = True
                break
        # If not in kwargs, try first positional arg
        if not input_found and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
            input_found = True
        # Call the original component's forward
        output = original_component(*args, **kwargs)

        # Handle tuple outputs from transformer components
        if isinstance(output, tuple):
            # Apply hook to first element (hidden states) and preserve the rest
            hooked_first = self.hook_out(output[0])
            output = (hooked_first,) + output[1:]
        else:
            # Pass output through hook_out
            output = self.hook_out(output)

        return output

    def __getattr__(self, name: str) -> Any:
        # Only called if attribute not found through normal lookup
        # First check if it's a module attribute (like hook_in, hook_out)
        if hasattr(self, "_modules") and name in self._modules:
            return self._modules[name]

        # Only try to resolve aliases if compatibility mode is enabled
        if self.compatibility_mode == True:
            # Check if this is a deprecated hook alias
            resolved_hook = resolve_alias(self, name, self.hook_aliases)
            if resolved_hook is not None:
                return resolved_hook

            # Check if this is a deprecated property alias
            resolved_property = resolve_alias(self, name, self.property_aliases)
            if resolved_property is not None:
                return resolved_property

        # Avoid recursion by checking if we're looking for original_component
        if name == "original_component":
            # This should not happen since original_component is a property
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if this is a submodule that should be registered as a PyTorch module
        # but hasn't been yet. This prevents PyTorch's add_module from failing.
        if name in self.submodules:
            # Don't delegate to original component for submodules
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Try to get from original_component if it exists
        original_component = self._modules.get("_original_component", None)
        if original_component is not None:
            try:
                name_split = name.split(".")

                if len(name_split) > 1:
                    current = getattr(original_component, name_split[0])
                    for part in name_split[1:]:
                        current = getattr(current, part)
                    return current
                else:
                    return getattr(original_component, name)
            except AttributeError:
                pass

        # If we get here, the attribute wasn't found anywhere
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, with passthrough to original component for compatibility."""
        # Handle normal PyTorch module attributes and our own attributes

        # Check if this is a HookPoint being set
        if isinstance(value, HookPoint):
            self._register_hook(name, value)
            super().__setattr__(name, value)
            return

        if name.startswith("_") or name in [
            "name",
            "config",
            "submodules",
            "conversion_rule",
            "compatibility_mode",
            "disable_warnings",
        ]:
            super().__setattr__(name, value)
            return

        # Check if this is a property on our class - if so, try to set it normally
        class_attr = getattr(type(self), name, None)
        if class_attr is not None and isinstance(class_attr, property):
            if class_attr.fset is not None:
                super().__setattr__(name, value)
                return
            # If it's a property with no setter, try the original component instead

        # Try to set the attribute on the original component if we have one
        if hasattr(self, "_modules") and "_original_component" in self._modules:
            original_component = self._modules["_original_component"]
            # Check if the attribute exists on the original component before setting
            if hasattr(original_component, name):
                try:
                    setattr(original_component, name, value)
                    return
                except AttributeError:
                    pass

        # Fall back to normal attribute setting
        super().__setattr__(name, value)
