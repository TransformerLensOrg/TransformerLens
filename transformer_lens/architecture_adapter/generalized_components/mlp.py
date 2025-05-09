"""MLP bridge component implementation."""

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.hook_points import HookPoint


class MLPBridge(GeneralizedComponent):
    """MLP bridge that wraps transformer MLP layers.
    
    This component provides hook points for:
    - Input to up projection
    - Up projection output
    - Down projection output
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the MLP bridge.
        
        Args:
            original_component: The original MLP component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
        # Initialize hook points
        self.hook_pre = HookPoint()  # Input to MLP
        self.hook_post = HookPoint()  # Final output
        
        # Set hook names
        self.hook_pre.name = f"{name}.pre"
        self.hook_post.name = f"{name}.post"
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP bridge.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Output hidden states
        """
        # Apply pre hook
        hidden_states = self.hook_pre(hidden_states)
        
        # Forward through original component
        output = self.original_component(hidden_states)
        
        # Apply post hook
        output = self.hook_post(output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output 