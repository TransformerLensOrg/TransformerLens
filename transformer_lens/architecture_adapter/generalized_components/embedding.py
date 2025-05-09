"""Embedding bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class EmbeddingBridge(GeneralizedComponent):
    """Embedding bridge that wraps transformer embedding layers.
    
    This component provides standardized hook points for:
    - input embeddings
    - position embeddings (if applicable)
    - final embedding output
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the embedding bridge.
        
        Args:
            original_component: The original embedding component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the embedding bridge.
        
        Args:
            input_ids: Input token IDs
            position_ids: Optional position IDs
            **kwargs: Additional arguments to pass to the original component
            
        Returns:
            Embedded output
        """
        # Execute pre-embedding hooks
        input_ids = self.execute_hooks("pre_embedding", input_ids)
        
        # Forward through original component
        output = self.original_component(input_ids, position_ids=position_ids, **kwargs)
        
        # Execute post-embedding hooks
        output = self.execute_hooks("post_embedding", output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output 