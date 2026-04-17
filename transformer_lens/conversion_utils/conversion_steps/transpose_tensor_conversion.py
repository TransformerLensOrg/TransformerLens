"""Transpose tensor conversion."""
import torch

from .base_tensor_conversion import BaseTensorConversion


class TransposeTensorConversion(BaseTensorConversion):
    """Transposes a 2D tensor.

    This conversion swaps the dimensions of a 2D tensor using .T

    Example:
        Input: [768, 50257]
        Output: [50257, 768]
    """

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Transpose the input tensor.

        Args:
            input_value: Input tensor to transpose
            *full_context: Additional context (unused)

        Returns:
            Transposed tensor
        """
        if not isinstance(input_value, torch.Tensor):
            return input_value

        if len(input_value.shape) != 2:
            # Only transpose 2D tensors
            return input_value

        return input_value.T

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        """Revert the transpose (transpose is its own inverse).

        Args:
            input_value: Input tensor to transpose
            *full_context: Additional context (unused)

        Returns:
            Transposed tensor
        """
        if not isinstance(input_value, torch.Tensor):
            return input_value

        if len(input_value.shape) != 2:
            # Only transpose 2D tensors
            return input_value

        return input_value.T

    def __repr__(self):
        return "TransposeTensorConversion()"
