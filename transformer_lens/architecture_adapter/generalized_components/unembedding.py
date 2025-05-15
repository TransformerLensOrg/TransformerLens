"""Unembedding bridge component.

This module contains the bridge component for unembedding layers.
"""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.hook_points import HookPoint


class UnembeddingBridge(GeneralizedComponent):
    """Bridge component for unembedding layers.
    
    This component wraps an unembedding layer from a remote model and provides a consistent interface
    for accessing its weights and performing unembedding operations.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: Any | None = None,
    ):
        """Initialize the unembedding bridge.
        
        Args:
            original_component: The original unembedding component to wrap
            name: The name of the component in the model
            architecture_adapter: The architecture adapter instance
        """
        super().__init__(original_component, name, architecture_adapter)
        
        # Initialize hook points
        self.hook_input = HookPoint()  # Input to projection
        self.hook_logits = HookPoint()  # Logits output
        
        # Set hook names
        self.hook_input.name = f"{name}.input"
        self.hook_logits.name = f"{name}.logits"
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the unembedding bridge.
        
        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component
            
        Returns:
            Logits output
        """
        # Apply input hook
        hidden_states = self.hook_input(hidden_states)
        
        # Forward through original component
        output = self.original_component(hidden_states, **kwargs)
        
        # Apply logits hook
        output = self.hook_logits(output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output

    @classmethod
    def wrap_component(cls, component: nn.Module, name: str, architecture_adapter: Any | None = None) -> nn.Module:
        """Wrap a component with this bridge if it's an unembedding layer.
        
        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance
            
        Returns:
            The wrapped component if it's an unembedding layer, otherwise the original component
        """
        if name.endswith(".unembed") or name.endswith(".lm_head"):
            return cls(component, name, architecture_adapter)
        return component 