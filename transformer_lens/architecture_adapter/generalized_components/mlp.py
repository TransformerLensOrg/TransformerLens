"""Generalized MLP component implementation."""

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class GeneralizedMLP(GeneralizedComponent):
    """Generalized MLP component that wraps transformer MLP layers.
    
    This component provides standardized hook points for:
    - input projection
    - activation
    - output projection
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the MLP component.
        
        Args:
            original_component: The original MLP component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP component.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Output hidden states
        """
        # Execute pre-MLP hooks
        hidden_states = self.execute_hooks("pre_mlp", hidden_states)
        
        # Forward through original component
        output = self.original_component(hidden_states)
        
        # Execute post-MLP hooks
        output = self.execute_hooks("post_mlp", output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output 