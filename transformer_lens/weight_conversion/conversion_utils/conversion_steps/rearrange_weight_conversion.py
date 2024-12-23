import einops
import torch

from .base_weight_conversion import BaseWeightConversion


class RearrangeWeightConversion(BaseWeightConversion):
    def __init__(self, pattern: str, **axes_lengths):
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def convert(self, input_value: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(input_value, self.pattern, **self.axes_lengths)
