"""Embedding bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.hook_points import HookPoint


class EmbeddingBridge(GeneralizedComponent):
    """Embedding bridge that wraps transformer embedding layers.
    
    This component provides hook points for:
    - Token embeddings
    - Position embeddings
    - Combined embeddings
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the embedding bridge.
        
        Args:
            original_component: The original embedding component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
        # Initialize hook points
        self.hook_embed = HookPoint()  # Token embeddings
        self.hook_pos = HookPoint()  # Position embeddings
        self.hook_output = HookPoint()  # Combined embeddings
        
        assert hasattr(self, "hook_embed")
        
        # Set hook names
        self.hook_embed.name = f"{name}.embed"
        self.hook_pos.name = f"{name}.pos"
        self.hook_output.name = f"{name}.output"
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the embedding bridge.
        
        Args:
            input_ids: Input token IDs
            position_ids: Optional position IDs (ignored if not supported)
            **kwargs: Additional arguments to pass to the original component
            
        Returns:
            Embedded output
        """
        # Forward through original component
        # Remove position_ids if not supported
        if not hasattr(self.original_component, "forward") or "position_ids" not in self.original_component.forward.__code__.co_varnames:
            kwargs.pop("position_ids", None)
            output = self.original_component(input_ids, **kwargs)
        else:
            output = self.original_component(input_ids, position_ids=position_ids, **kwargs)
        
        # Apply hook to final output
        output = self.hook_output(output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "output": output
        })
        
        return output 