"""Attention Auto Conversion

This module provides automatic conversion for attention hook inputs.
It handles conversions for attention activation tensors flowing through hooks.
For most cases, attention hooks don't require conversion of the activation tensors.
"""

from typing import Any

from .base_hook_conversion import BaseHookConversion


class AttentionAutoConversion(BaseHookConversion):
    """Handles conversions for attention hook inputs (activation tensors)."""
    
    def __init__(self, config: Any):
        """Initialize the attention auto conversion.
        
        Args:
            config: Model configuration (available for future use if needed)
        """
        super().__init__()
        self.config = config
        
    def handle_conversion(self, input_value: Any, *full_context) -> Any:
        """Reshape hook tensor to match HookedTransformer format.
        
        Only handles the specific tensor passed into the hook to ensure it matches
        the shape expected by HookedTransformer. Main focus is attention patterns.
        
        Args:
            input_value: The tensor input (activation) flowing through the hook
            *full_context: Additional context (not used)
            
        Returns:
            The tensor reshaped to match HookedTransformer expectations
        """
        import einops
        import torch
        
        if not isinstance(input_value, torch.Tensor):
            return input_value
        
        # Get model dimensions if available
        n_heads = getattr(self.config, 'n_heads', None) or getattr(self.config, 'num_attention_heads', None)
        
        # Main case: 4D attention patterns - ensure (batch, head_index, query_pos, key_pos) format
        if len(input_value.shape) == 4:
            batch, dim1, dim2, dim3 = input_value.shape
            
            # Case 1: (batch, query_pos, head_index, key_pos) -> (batch, head_index, query_pos, key_pos)
            if n_heads and dim2 == n_heads and dim1 == dim3:
                return einops.rearrange(input_value, "batch query_pos head_index key_pos -> batch head_index query_pos key_pos")
                
            # Case 2: Already correct (batch, head_index, query_pos, key_pos)
            elif n_heads and dim1 == n_heads and dim2 == dim3:
                return input_value
                
            # Case 3: Try simple transpose for square matrices
            elif dim1 == dim3 and dim2 == dim3:
                return input_value.transpose(1, 2)
        
        # For all other cases (3D, 2D, 1D), pass through unchanged
        # The hook is receiving the tensor in the right format already
        return input_value
        
    def __repr__(self) -> str:
        """String representation of the conversion."""
        return f"AttentionAutoConversion(config={self.config})"