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
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass through the attention bridge.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional past key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            position_embeddings: Optional position embeddings
            **kwargs: Additional arguments to pass to the original component
            
        Returns:
            The output from the original component, with hooks applied
        """
        # Forward through original component
        outputs = self.original_component(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # Always get attention weights for hooks
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # Define which outputs should go through which hooks
        hook_map = {
            0: "output",  # First output (hidden_states) goes through output hook
            1: "attn",    # Second output (attention_weights) goes through attn hook
        }
        
        # Apply hooks to outputs
        outputs = self._apply_hooks_to_outputs(outputs, hook_map)
        
        # Store hook outputs
        if isinstance(outputs, tuple):
            self.hook_outputs.update({
                "output": outputs[0],
                "attention_weights": outputs[1] if len(outputs) > 1 else None
            })
        else:
            self.hook_outputs.update({
                "output": outputs
            })
        
        # Return the outputs in the same format as the original component
        return outputs 