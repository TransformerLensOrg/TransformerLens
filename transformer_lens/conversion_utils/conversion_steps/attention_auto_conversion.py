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
        """Apply conversions to attention tensor inputs.
        
        Args:
            input_value: The tensor input (activation) flowing through the hook
            *full_context: Additional context (not used)
            
        Returns:
            The input_value unchanged (no conversion needed for attention activations)
        """
        # For attention hooks, we typically don't need to convert activation tensors
        # The activations are already in the correct format from the original model
        # If specific tensor transformations are needed in the future, they can be added here
        return input_value
        
    def __repr__(self) -> str:
        """String representation of the conversion."""
        return f"AttentionAutoConversion(config={self.config})"