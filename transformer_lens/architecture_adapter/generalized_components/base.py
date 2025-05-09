"""Base class for generalized transformer components."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.
    
    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the generalized component.
        
        Args:
            original_component: The original transformer component to wrap
            name: The name of this component
        """
        super().__init__()
        self.original_component = original_component
        self.name = name
        self.hooks: dict[str, list[Callable]] = {}
        self.hook_outputs: dict[str, Any] = {}

    def register_hook(self, hook_name: str, hook_fn: Callable) -> None:
        """Register a hook function for a specific hook point.
        
        Args:
            hook_name: Name of the hook point
            hook_fn: Function to call at this hook point
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(hook_fn)

    def remove_hook(self, hook_name: str, hook_fn: Callable) -> None:
        """Remove a previously registered hook.
        
        Args:
            hook_name: Name of the hook point
            hook_fn: Function to remove
        """
        if hook_name in self.hooks:
            self.hooks[hook_name].remove(hook_fn)

    def execute_hooks(self, hook_name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Execute all hooks registered for a specific hook point.
        
        Args:
            hook_name: Name of the hook point
            tensor: The tensor to pass through the hooks
            
        Returns:
            The result of the last hook execution, or the input tensor if no hooks
        """
        # Store the input tensor as the hook output
        self.hook_outputs[hook_name] = tensor
        
        # Execute hooks if any
        if hook_name in self.hooks:
            result = tensor
            for hook in self.hooks[hook_name]:
                hook_result = hook(result)
                if hook_result is not None:
                    result = hook_result
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

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass through the component.
        
        This should be implemented by subclasses to define the specific
        behavior of the component.
        
        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments
            
        Returns:
            Either a single tensor or a tuple of tensors
        """
        raise NotImplementedError("Subclasses must implement forward()") 