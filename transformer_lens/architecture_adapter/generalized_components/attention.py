"""Generalized attention component implementation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class GeneralizedAttention(GeneralizedComponent):
    """Generalized attention component that wraps transformer attention layers.
    
    This component provides standardized hook points for:
    - query/key/value projections
    - attention scores
    - attention output
    - final output projection
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the attention component.
        
        Args:
            original_component: The original attention component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the attention component.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional past key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            
        Returns:
            Tuple containing:
            - Output hidden states
            - Optional attention weights
            - Optional present key/value states
        """
        # Execute pre-attention hooks
        hidden_states = self.execute_hooks("pre_attention", hidden_states) or hidden_states
        
        # Forward through original component
        outputs = self.original_component(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # Always get attention weights for hooks
            use_cache=use_cache,
        )
        
        # Unpack outputs
        output = outputs[0]
        attention_weights = outputs[1] if len(outputs) > 1 else None
        present_key_value = outputs[2] if len(outputs) > 2 else None
        
        # Execute post-attention hooks
        output = self.execute_hooks("post_attention", output) or output
        
        # Store hook outputs
        self.hook_outputs.update({
            "attention_weights": attention_weights,
            "output": output
        })
        
        # Return appropriate outputs based on flags
        outputs = (output,)
        if output_attentions and attention_weights is not None:
            outputs += (attention_weights,)
        if use_cache and present_key_value is not None:
            outputs += (present_key_value,)
            
        return outputs 