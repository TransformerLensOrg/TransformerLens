"""Base class for generalized transformer components."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.

    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    # Class attribute indicating whether this component represents a list item (like blocks)
    is_list_item: bool = False

    # Dictionary mapping deprecated hook names to their new equivalents
    # Subclasses can override this to define their own aliases
    hook_aliases: Dict[str, str] = {}

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,
    ):
        """Initialize the generalized component.

        Args:
            name: The name of this component
            config: Optional configuration object for the component
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__()
        self.name = name
        self.config = config
        self.submodules = submodules or {}

        # Standardized hooks for all bridge components
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

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

    def __getattr__(self, name: str):
        # Only called if attribute not found through normal lookup
        # First check if it's a module attribute (like hook_in, hook_out)
        if hasattr(self, "_modules") and name in self._modules:
            return self._modules[name]

        # Check if this is a deprecated hook alias
        if name in self.hook_aliases:
            target_hook = self.hook_aliases[name]
            warnings.warn(
                f"Hook '{name}' is deprecated and will be removed in a future version. "
                f"Use '{target_hook}' instead.",
                FutureWarning,
                stacklevel=2,
            )
            # Return the target hook
            return getattr(self, target_hook)

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
                return getattr(original_component, name)
            except AttributeError:
                pass

        # If we get here, the attribute wasn't found anywhere
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
