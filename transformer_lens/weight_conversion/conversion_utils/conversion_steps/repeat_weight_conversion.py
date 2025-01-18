import einops
import torch

from .base_weight_conversion import BaseWeightConversion


class RepeatWeightConversion(BaseWeightConversion):
    def __init__(self, pattern: str, input_filter: callable|None = None, output_filter: callable|None = None, **axes_lengths):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def handle_conversion(self, input_value: torch.Tensor) -> torch.Tensor:
        return einops.repeat(input_value, self.pattern, **self.axes_lengths)
