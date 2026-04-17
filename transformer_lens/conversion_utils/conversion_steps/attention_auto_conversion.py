"""Attention Auto Conversion

This module provides automatic conversion for attention hook inputs with revert capability.
It handles bidirectional conversions for attention activation tensors flowing through hooks.
"""

from typing import Any, Dict, Optional

import einops
import torch

from .base_tensor_conversion import BaseTensorConversion


class AttentionAutoConversion(BaseTensorConversion):
    """Handles bidirectional conversions for attention hook inputs (activation tensors).

    Converts tensors to match HookedTransformer format and can revert them back
    to their original format using stored state information.
    """

    def __init__(self, config: Any):
        """Initialize the attention auto conversion.

        Args:
            config: Model configuration containing attention head information
        """
        super().__init__()
        self.config = config
        self._conversion_state: Dict[int, Dict[str, Any]] = {}

    def handle_conversion(self, input_value: Any, *full_context) -> Any:
        """Convert tensor to HookedTransformer format and store revert state.

        Args:
            input_value: The tensor input (activation) flowing through the hook
            *full_context: Additional context (not used)

        Returns:
            The tensor reshaped to match HookedTransformer expectations
        """
        if not isinstance(input_value, torch.Tensor):
            return input_value

        tensor_id = id(input_value)
        original_shape = input_value.shape
        n_heads = getattr(self.config, "n_heads", None) or getattr(
            self.config, "num_attention_heads", None
        )

        # Store original state for revert
        self._conversion_state[tensor_id] = {
            "original_shape": original_shape,
            "conversion_type": None,
            "n_heads": n_heads,
        }

        # Handle 4D attention patterns - ensure (batch, head_index, query_pos, key_pos) format
        if len(original_shape) == 4:
            batch, dim1, dim2, dim3 = original_shape

            # Case 1: (batch, query_pos, head_index, key_pos) -> (batch, head_index, query_pos, key_pos)
            if n_heads and dim2 == n_heads and dim1 == dim3:
                self._conversion_state[tensor_id]["conversion_type"] = "transpose_1_2"
                return einops.rearrange(
                    input_value,
                    "batch query_pos head_index key_pos -> batch head_index query_pos key_pos",
                )

            # Case 2: Already correct (batch, head_index, query_pos, key_pos)
            elif n_heads and dim1 == n_heads and dim2 == dim3:
                self._conversion_state[tensor_id]["conversion_type"] = "no_change"
                return input_value

            # Case 3: Simple transpose for square matrices
            elif dim1 == dim3 and dim2 == dim3:
                self._conversion_state[tensor_id]["conversion_type"] = "transpose_1_2"
                return input_value.transpose(1, 2)

        # No conversion needed
        self._conversion_state[tensor_id]["conversion_type"] = "no_change"
        return input_value

    def revert_conversion(
        self, converted_value: Any, original_tensor_id: Optional[int] = None
    ) -> Any:
        """Revert tensor back to its original format using stored state.

        Args:
            converted_value: The tensor that was previously converted
            original_tensor_id: ID of the original tensor (if available)

        Returns:
            The tensor reverted to its original format
        """
        if not isinstance(converted_value, torch.Tensor):
            return converted_value

        # Try to find conversion state
        tensor_id = original_tensor_id or id(converted_value)
        state = self._conversion_state.get(tensor_id)

        if state is None:
            # No stored state, return as-is
            return converted_value

        conversion_type = state["conversion_type"]

        # Apply reverse conversion based on stored type
        if conversion_type == "transpose_1_2":
            # Reverse the transpose operation
            if len(converted_value.shape) == 4:
                return converted_value.transpose(1, 2)
        elif conversion_type == "no_change":
            return converted_value

        return converted_value

    def clear_state(self, tensor_id: Optional[int] = None) -> None:
        """Clear stored conversion state.

        Args:
            tensor_id: Specific tensor ID to clear, or None to clear all
        """
        if tensor_id is not None:
            self._conversion_state.pop(tensor_id, None)
        else:
            self._conversion_state.clear()

    def get_conversion_info(self, tensor_id: int) -> Optional[Dict[str, Any]]:
        """Get conversion information for a tensor.

        Args:
            tensor_id: ID of the tensor to get info for

        Returns:
            Dictionary with conversion information or None if not found
        """
        return self._conversion_state.get(tensor_id)

    def __repr__(self) -> str:
        """String representation of the conversion."""
        return f"AttentionAutoConversion(config={self.config}, active_states={len(self._conversion_state)})"
