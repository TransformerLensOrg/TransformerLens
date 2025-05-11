"""Attention bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.hook_points import HookPoint


class AttentionBridge(GeneralizedComponent):
    """Attention bridge that wraps transformer attention layers.
    
    This component provides hook points for:
    - Query/key/value projections
    - Attention scores
    - Attention output
    """

    def __init__(self, original_component: nn.Module, name: str):
        """Initialize the attention bridge.
        
        Args:
            original_component: The original attention component to wrap
            name: The name of this component
        """
        super().__init__(original_component, name)
        
        # Initialize hook points
        self.hook_q_input = HookPoint()  # Input to query projection
        self.hook_k_input = HookPoint()  # Input to key projection
        self.hook_v_input = HookPoint()  # Input to value projection
        
        self.hook_q = HookPoint()  # Query projection output
        self.hook_k = HookPoint()  # Key projection output
        self.hook_v = HookPoint()  # Value projection output
        
        self.hook_pattern = HookPoint()  # Raw attention scores
        self.hook_attn = HookPoint()  # Normalized attention scores
        
        self.hook_z = HookPoint()  # Attention output before projection
        self.hook_output = HookPoint()  # Final output after projection
        
        # Set hook names
        self.hook_q_input.name = f"{name}.q_input"
        self.hook_k_input.name = f"{name}.k_input"
        self.hook_v_input.name = f"{name}.v_input"
        self.hook_q.name = f"{name}.q"
        self.hook_k.name = f"{name}.k"
        self.hook_v.name = f"{name}.v"
        self.hook_pattern.name = f"{name}.pattern"
        self.hook_attn.name = f"{name}.attn"
        self.hook_z.name = f"{name}.z"
        self.hook_output.name = f"{name}.output"
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            - Optional attention weights
        """
        # Forward through original component
        outputs = self.original_component(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # Always get attention weights for hooks
            use_cache=use_cache,
            **kwargs,
        )
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            if len(outputs) == 2:  # (hidden_states, attention_weights)
                output, attention_weights = outputs
            else:  # (hidden_states, attention_weights, present_key_value)
                output = outputs[0]
                attention_weights = outputs[1]
        else:  # Just hidden_states
            output = outputs
            attention_weights = None
        
        # Apply hooks
        if attention_weights is not None:
            attention_weights = self.hook_attn(attention_weights)
            
        output = self.hook_output(output)
        
        # Store hook outputs
        self.hook_outputs.update({
            "attention_weights": attention_weights,
            "output": output
        })
        
        # Return just hidden_states and attention_weights
        return output, attention_weights if output_attentions else None 