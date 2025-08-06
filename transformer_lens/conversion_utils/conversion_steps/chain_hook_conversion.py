"""Chain weight conversion step."""
from typing import List

from torch import Tensor

from .base_hook_conversion import BaseHookConversion


class ChainHookConversion(BaseHookConversion):
    """Chain multiple weight conversion steps together."""

    def __init__(self, conversions: List[BaseHookConversion]):
        """Initialize the ChainHookConversion.

        Args:
            conversions (List[BaseHookConversion]): A list of conversions to apply in order.
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
