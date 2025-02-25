import torch

from .base_weight_conversion import BaseWeightConversion


class ZerosLikeConversion(BaseWeightConversion):
    def handle_conversion(self, input_value: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input_value)
