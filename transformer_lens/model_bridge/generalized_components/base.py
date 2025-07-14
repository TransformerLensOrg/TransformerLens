"""Base class for generalized transformer components."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint

if TYPE_CHECKING:
    pass


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.

    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    # Class attribute indicating whether this component represents a list item (like blocks)
    is_list_item: bool = False

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
        # Use object.__setattr__ to avoid PyTorch's module system
        object.__setattr__(self, "_original_component", None)
        self.hooks: dict[str, list[Callable[..., torch.Tensor]]] = {}
        self.hook_outputs: dict[str, Any] = {}
        self._hook_tracker = None

        # Standardized hooks for all bridge components - use add_module to ensure proper registration
        self.add_module("hook_in", HookPoint())
        self.add_module("hook_out", HookPoint())

    def set_original_component(self, original_component: nn.Module) -> None:
        """Set the original component that this bridge wraps.

        Args:
            original_component: The original transformer component to wrap
        """
        # Use object.__setattr__ to avoid PyTorch's module system
        object.__setattr__(self, "_original_component", original_component)

    @property
    def original_component(self) -> Optional[nn.Module]:
        """Get the original component."""
        return object.__getattribute__(self, "_original_component")

    def set_hook_tracker(self, tracker: Any) -> None:
        """Set the hook tracker for this component.

        Args:
            tracker: The hook tracker instance
        """
        self._hook_tracker = tracker

    def get_hook_tracker(self) -> Any | None:
        """Get the hook tracker for this component.

        Returns:
            The hook tracker instance if set, None otherwise
        """
        return self._hook_tracker

    def register_hook(self, hook_name: str, hook_fn: Callable[..., torch.Tensor]) -> None:
        """Register a hook function for a specific hook point.

        Args:
            hook_name: Name of the hook point
            hook_fn: Function to call at this hook point
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(hook_fn)

        # If we have a hook tracker, register the hook there too
        if self._hook_tracker is not None:
            self._hook_tracker.register_component_hook(self.name, hook_name, hook_fn)

    def remove_hook(self, hook_name: str, hook_fn: Callable[..., torch.Tensor]) -> None:
        """Remove a previously registered hook.

        Args:
            hook_name: Name of the hook point
            hook_fn: Function to remove
        """
        if hook_name in self.hooks:
            self.hooks[hook_name].remove(hook_fn)
            if not self.hooks[hook_name]:  # If no hooks left, remove the entry
                del self.hooks[hook_name]

    def execute_hooks(self, hook_name: str, tensor: torch.Tensor | tuple) -> torch.Tensor | tuple:
        """Execute all hooks registered for a specific hook point.

        Args:
            hook_name: Name of the hook point
            tensor: The tensor or tuple to pass through the hooks

        Returns:
            The result of the last hook execution, or the input tensor if no hooks
        """
        # Store the input tensor as the hook output
        self.hook_outputs[hook_name] = tensor

        # Execute hooks if any
        if hook_name in self.hooks:
            result = tensor
            for hook in self.hooks[hook_name]:
                # For tuple outputs (like attention), pass the first element to hooks
                if isinstance(result, tuple):
                    hooked_first = hook(result[0], hook=self)  # Pass hook object as second argument
                    result = (hooked_first,) + result[1:]
                else:
                    result = hook(result, hook=self)  # Pass hook object as second argument
                # Update the hook output with the modified tensor
                self.hook_outputs[hook_name] = result
            return result

        return tensor

    def get_hook_output(self, hook_name: str) -> Any:
        """Get the output from a specific hook point.

        Args:
            hook_name: Name of the hook point

        Returns:
            The stored output for this hook point
        """
        return self.hook_outputs.get(hook_name)

    def clear_hooks(self) -> None:
        """Clear all registered hooks."""
        self.hooks.clear()
        self.hook_outputs.clear()

    def add_hook(self, hook_fn: Callable[..., torch.Tensor], hook_name: str = "output") -> None:
        """Add a hook function (HookedTransformer-compatible interface).

        Args:
            hook_fn: Function to call at this hook point
            hook_name: Name of the hook point (defaults to "output")
        """
        self.register_hook(hook_name, hook_fn)

    def remove_hooks(self, hook_name: str | None = None) -> None:
        """Remove hooks (HookedTransformer-compatible interface).

        Args:
            hook_name: Name of the hook point to remove. If None, removes all hooks.
        """
        if hook_name is None:
            self.clear_hooks()
        else:
            if hook_name in self.hooks:
                del self.hooks[hook_name]
            if hook_name in self.hook_outputs:
                del self.hook_outputs[hook_name]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Generic forward pass for bridge components with input/output hooks."""
        original_component = object.__getattribute__(self, "_original_component")
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
        # Pass output through hook_out
        output = self.hook_out(output)
        self.hook_outputs.update({"output": output})
        return output

    def _apply_hooks_to_outputs(self, outputs: Any, hook_map: dict[str | int, str]) -> Any:
        """Apply hooks to outputs from the original component.

        Args:
            outputs: The outputs from the original component
            hook_map: Dictionary mapping output indices/keys to hook names

        Returns:
            The outputs with hooks applied
        """
        if isinstance(outputs, tuple):
            # For tuple outputs, apply hooks to each element that has a hook
            return tuple(
                self.execute_hooks(hook_map[i], out) if i in hook_map else out
                for i, out in enumerate(outputs)
            )
        elif isinstance(outputs, dict):
            # For dict outputs, apply hooks to each value that has a hook
            return {
                k: self.execute_hooks(hook_map[k], v) if k in hook_map else v
                for k, v in outputs.items()
            }
        else:
            # For single tensor outputs, apply the default hook
            return self.execute_hooks(hook_map.get("output", "output"), outputs)

    def __getattr__(self, name: str):
        # Only called if attribute not found through normal lookup
        # First check if it's a module attribute (like hook_in, hook_out)
        if hasattr(self, "_modules") and name in self._modules:
            return self._modules[name]

        # Avoid recursion by checking if we're looking for original_component
        if name == "original_component":
            # This should not happen since original_component is a property
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Try to get from original_component if it exists
        try:
            original_component = object.__getattribute__(self, "_original_component")
            if original_component is not None:
                try:
                    return getattr(original_component, name)
                except AttributeError:
                    pass
        except AttributeError:
            # _original_component doesn't exist
            pass

        # If we get here, the attribute wasn't found anywhere
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
