"""Base class for generalized transformer components."""

from __future__ import annotations

import dis
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

        # Standardized hooks for all bridge components
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

        # Apply conversion rule to hooks if available
        if self.conversion_rule is not None:
            self.hook_in.hook_conversion = self.conversion_rule
            self.hook_out.hook_conversion = self.conversion_rule

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

    def _getattr_helper(self, name: str) -> Any:
        """This function contains the main getattr logic for the component.
        It is extracted into a helper function to avoid recursion issues when trying
        to access certain aliased attributes like W_Q, W_K, W_V, etc."""

        # Only called if attribute not found through normal lookup
        # First check if it's a module attribute (like hook_in, hook_out)
        if hasattr(self, "_modules") and name in self._modules:
            return self._modules[name]

        # Check if this is a deprecated hook alias
        resolved_hook = resolve_alias(self, name, self.hook_aliases)
        if resolved_hook is not None and self.compatibility_mode == True:
            return resolved_hook

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

    def __getattr__(self, name: str):
        # This code is to check if __getattr__ was called internally, because we only want to resolve aliases
        # if the user is trying to access an attribute directly, not if it's being called internally during setup or run_with_cache.

        # Get execution frame of caller
        current_frame = inspect.currentframe()
        if current_frame is None:
            # If we can't get frame info, fall back to regular attribute access
            return self._getattr_helper(name)

        frame = current_frame.f_back
        # Extract the module name from the frame
        caller_module = frame.f_globals.get("__name__", "") if frame else ""

        # Check 1: Is __getattr__ being called internally
        if not caller_module.startswith("transformer_lens") and not caller_module.startswith(
            "torch"
        ):
            # Check 2: Is next access .weight or .bias?
            # If the user is correctly accessing a property like W_Q.weight or W_Q.bias,
            # we want to return the original W_Q and not W_Q.weight (the alias), because otherwise
            # we would essentially be calling W_Q.weight.weight which causes an error.
            if frame is not None:
                try:
                    # Get bytecode instructions of the current frame
                    instructions = list(dis.get_instructions(frame.f_code))
                    for instr in instructions:
                        # Find next instruction after the current one (frame.f_lasti)
                        if instr.offset > frame.f_lasti:
                            # If the next instruction is a LOAD_ATTR and the attribute is weight or bias,
                            # we want to return the original W_Q, not W_Q.weight or W_Q.bias
                            if instr.opname == "LOAD_ATTR" and instr.argval in ["weight", "bias"]:
                                return self._getattr_helper(name)
                            break
                except (AttributeError, ValueError, TypeError):
                    pass

            # If we reach here, we can resolve the alias normally
            resolved_property = resolve_alias(self, name, self.property_aliases)

            if resolved_property is not None and self.compatibility_mode == True:
                return resolved_property

        # If an internal call or no alias was found, just regularly get the attribute
        return self._getattr_helper(name)
