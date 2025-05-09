"""Unembedding bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.hook_points import HookPoint


class UnembeddingBridge(GeneralizedComponent):
    """Unembedding bridge that wraps transformer language model heads.
    
    This component provides hook points for:
    - Input to projection
    - Logits output
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the unembedding bridge.
        
        Args:
            original_component: The original unembedding component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
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