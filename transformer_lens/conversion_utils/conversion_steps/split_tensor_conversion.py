"""Split weight conversion step."""
import torch
from torch import Tensor

from .base_tensor_conversion import BaseTensorConversion


class SplitTensorConversion(BaseTensorConversion):
    """Split a weight tensor along a specified dimension."""

    def __init__(self, index: int, num_splits: int, dim: int = 0):
        """Initialize the SplitTensorConversion.

        Args:
            index (int): The index of the split to select.
            num_splits (int): The total number of splits.
            dim (int, optional): The dimension to split along. Defaults to 0.
        """
        super().__init__()
        self.index = index
        self.num_splits = num_splits
        self.dim = dim

    def handle_conversion(self, input_value: Tensor, *full_context) -> Tensor:
        """Convert the weight by splitting it and selecting a chunk.

        Args:
            input_value (torch.Tensor): The weight to convert.

        Returns:
            torch.Tensor: The converted weight.
        """
        chunks = torch.chunk(input_value, self.num_splits, dim=self.dim)
        return chunks[self.index]
