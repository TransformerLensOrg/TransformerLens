import torch

from .base_weight_conversion import BaseWeightConversion


class ZerosLikeConversion(BaseWeightConversion):
    def handle_conversion(self, input_value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input_value)

    def __repr__(self):
        return f"Is a zeros_like operation resulting in a Tensor filled with zeros of shape: {self.input_value.shape}"
