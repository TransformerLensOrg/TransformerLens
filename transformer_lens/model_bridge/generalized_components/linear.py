"""Linear bridge component for wrapping linear layers with hook points."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LinearBridge(GeneralizedComponent):
    """Bridge component for linear layers.
    
    This component wraps a linear layer (nn.Linear) and provides hook points
    for intercepting the input and output activations.
    """

    def __init__(self, original_component: nn.Linear, name: str, architecture_adapter: Any, **kwargs: Any) -> None:
        """Initialize the LinearBridge.
        
        Args:
            original_component: The original nn.Linear layer to wrap
            name: The name of this component
            architecture_adapter: Architecture adapter for component-specific operations
            **kwargs: Additional keyword arguments
        """
        super().__init__(original_component, name, architecture_adapter)
        
        # Store linear layer properties for easy access
        self.in_features = original_component.in_features
        self.out_features = original_component.out_features
        self.bias = original_component.bias is not None

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the linear layer with hooks.
        
        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor after linear transformation
        """
        # Apply input hook
        input = self.hook_in(input)
        
        # Forward through the original linear layer
        output = self.original_component(input, *args, **kwargs)
        
        # Apply output hook
        output = self.hook_out(output)
        
        return output

    def __repr__(self) -> str:
        """String representation of the LinearBridge."""
        return f"LinearBridge({self.in_features} -> {self.out_features}, bias={self.bias})" 