"""Layer norm bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class LayerNormBridge(GeneralizedComponent):
    """Layer norm bridge that wraps transformer layer normalization layers.
    
    This component provides standardized hook points for:
    - input normalization
    - scale factor
    - normalized output
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the layer norm bridge.
        
        Args:
            original_component: The original layer norm component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
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
        # Execute pre-norm hooks
        hidden_states = self.execute_hooks("pre_norm", hidden_states)
        
        # Forward through original component
        output = self.original_component(hidden_states, **kwargs)
        
        # Execute post-norm hooks
        output = self.execute_hooks("post_norm", output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output 