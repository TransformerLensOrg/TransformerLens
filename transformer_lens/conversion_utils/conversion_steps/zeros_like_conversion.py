import torch

from .base_tensor_conversion import BaseTensorConversion


class ZerosLikeConversion(BaseTensorConversion):
    def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
        return torch.zeros_like(input_value)

    def __repr__(self):
        return "Is a zeros_like operation"
