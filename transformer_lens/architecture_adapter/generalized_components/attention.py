"""Attention bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionBridge(GeneralizedComponent):
    """Attention bridge that wraps transformer attention layers.
    
    This component provides standardized hook points for:
    - query/key/value projections
    - attention scores
    - attention output
    - final output projection
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the attention bridge.
        
        Args:
            original_component: The original attention component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, ...] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,  # Accept any additional arguments
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...] | None]:
        """Forward pass through the attention bridge.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional past key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            **kwargs: Additional arguments to pass to the original component
            
        Returns:
            Tuple containing:
            - Output hidden states
            - Attention weights (None if not requested)
            - Present key/value states (None if not using cache)
        """
        # Execute pre-attention hooks
        hidden_states = self.execute_hooks("pre_attention", hidden_states)
        
        # Forward through original component
        outputs = self.original_component(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # Always get attention weights for hooks
            use_cache=use_cache,
            **kwargs,  # Pass through any additional arguments
        )
        
        # Unpack outputs
        output = outputs[0]
        attention_weights = outputs[1] if len(outputs) > 1 else None
        present_key_value = outputs[2] if len(outputs) > 2 else None
        
        # Execute post-attention hooks
        output = self.execute_hooks("post_attention", output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "attention_weights": attention_weights,
            "output": output
        })
        
        # Always return a 3-tuple with the expected types
        return (
            output,
            attention_weights if output_attentions else None,
            present_key_value if use_cache else None,
        ) 