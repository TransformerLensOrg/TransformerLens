from collections.abc import Callable
from typing import Optional

import einops
import torch

from .base_tensor_conversion import BaseTensorConversion


class RearrangeTensorConversion(BaseTensorConversion):
    def __init__(
        self,
        pattern: str,
        input_filter: Optional[Callable] = None,
        output_filter: Optional[Callable] = None,
        **axes_lengths,
    ):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        return einops.rearrange(input_value, self.pattern, **self.axes_lengths)

    def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        left, right = self.pattern.split("->")
        pattern = f"{right.strip()} -> {left.strip()}"
        return einops.rearrange(input_value, pattern, **self.axes_lengths)

    def __repr__(self):
        return f'Is a rearrange operation with the pattern "{self.pattern}"'
