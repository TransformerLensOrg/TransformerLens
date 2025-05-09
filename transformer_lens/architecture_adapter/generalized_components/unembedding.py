"""Unembedding bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class UnembeddingBridge(GeneralizedComponent):
    """Unembedding bridge that wraps transformer language model heads.
    
    This component provides standardized hook points for:
    - input projection
    - logits output
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the unembedding bridge.
        
        Args:
            original_component: The original unembedding component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
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
        # Execute pre-unembedding hooks
        hidden_states = self.execute_hooks("pre_unembedding", hidden_states)
        
        # Forward through original component
        output = self.original_component(hidden_states, **kwargs)
        
        # Execute post-unembedding hooks
        output = self.execute_hooks("post_unembedding", output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output 