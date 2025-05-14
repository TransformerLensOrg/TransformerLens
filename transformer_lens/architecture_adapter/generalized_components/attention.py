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
        
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the attention bridge, passing all arguments through."""
        outputs = self.original_component(*args, **kwargs)

        # Dynamically apply hooks to outputs if tuple or dict
        if isinstance(outputs, tuple):
            outputs = list(outputs)
            if len(outputs) > 0:
                outputs[0] = self.hook_output(outputs[0])
            if len(outputs) > 1:
                outputs[1] = self.hook_attn(outputs[1])
            outputs = tuple(outputs)
        elif isinstance(outputs, dict):
            if 'hidden_states' in outputs:
                outputs['hidden_states'] = self.hook_output(outputs['hidden_states'])
            if 'attn_weights' in outputs:
                outputs['attn_weights'] = self.hook_attn(outputs['attn_weights'])
        else:
            outputs = self.hook_output(outputs)

        return outputs 