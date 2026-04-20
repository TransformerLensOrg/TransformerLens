"""Chain weight conversion step."""
from typing import List

from torch import Tensor

from .base_tensor_conversion import BaseTensorConversion


class ChainTensorConversion(BaseTensorConversion):
    """Chain multiple weight conversion steps together."""

    def __init__(self, conversions: List[BaseTensorConversion]):
        """Initialize the ChainTensorConversion.

        Args:
            conversions (List[BaseTensorConversion]): A list of conversions to apply in order.
        """
        super().__init__()
        self.conversions = conversions

    def handle_conversion(self, input_value: Tensor, *full_context) -> Tensor:
        """Convert the weight by applying a chain of conversions.

        Args:
            input_value (torch.Tensor): The weight to convert.

        Returns:
            torch.Tensor: The converted weight.
        """
        for conversion in self.conversions:
            input_value = conversion.handle_conversion(input_value, *full_context)
        return input_value

    def revert(self, input_value: Tensor, *full_context) -> Tensor:
        """Revert the weight by applying conversions in reverse order.

        Args:
            input_value (torch.Tensor): The weight to revert.

        Returns:
            torch.Tensor: The reverted weight.
        """
        for conversion in reversed(self.conversions):
            input_value = conversion.revert(input_value, *full_context)
        return input_value
