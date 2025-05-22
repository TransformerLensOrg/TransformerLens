"""Layer norm bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.hook_points import HookPoint


class LayerNormBridge(GeneralizedComponent):
    """Layer norm bridge that wraps transformer layer normalization layers.
    
    This component provides hook points for:
    - Input to normalization
    - Scale factor
    - Normalized output
    """

    def __init__(self, original_component: nn.Module, name: str, architecture_adapter: Any | None = None):
        """Initialize the layer norm bridge.
        
        Args:
            original_component: The original layer norm component to wrap
            name: The name of this component
            architecture_adapter: Optional architecture adapter for component-specific operations
        """
        super().__init__(original_component, name, architecture_adapter)
        
        # Initialize hook points
        self.hook_scale = HookPoint()  # Scale factor
        self.hook_normalized = HookPoint()  # Normalized output
        
        # Set hook names
        self.hook_scale.name = f"{name}.scale"
        self.hook_normalized.name = f"{name}.normalized"
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the layer norm bridge.
        
        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component
            
        Returns:
            Normalized output
        """
        # Forward through original component
        output = self.original_component(hidden_states, **kwargs)
        
        # Apply hook to normalized output
        output = self.hook_normalized(output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output

    @classmethod
    def wrap_component(cls, component: nn.Module, name: str, architecture_adapter: Any | None = None) -> nn.Module:
        """Wrap a component with this bridge if it's a LayerNorm layer.
        
        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance
            
        Returns:
            The wrapped component if it's a LayerNorm layer, otherwise the original component
        """
        if name.endswith(".ln") or name.endswith(".ln1") or name.endswith(".ln2"):
            return cls(component, name, architecture_adapter)
        return component 