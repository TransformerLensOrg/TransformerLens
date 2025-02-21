import einops
from collections.abc import Callable
import torch
from typing import Optional

from .base_weight_conversion import BaseWeightConversion


class RepeatWeightConversion(BaseWeightConversion):
    def __init__(self, pattern: str, input_filter: Optional[Callable] = None, output_filter: Optional[Callable] = None, **axes_lengths):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def handle_conversion(self, input_value: torch.Tensor) -> torch.Tensor:
        return einops.repeat(input_value, self.pattern, **self.axes_lengths)
