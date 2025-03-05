import torch

from .base_weight_conversion import BaseWeightConversion


class ZerosLikeConversion(BaseWeightConversion):
    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        return torch.zeros_like(input_value)

    def __repr__(self):
        return f"Is a zeros_like operation"
