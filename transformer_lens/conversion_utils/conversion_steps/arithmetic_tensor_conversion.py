"""Weight conversion that performs arithmetic operations on weights."""

from collections.abc import Callable
from enum import Enum
from typing import Optional

import torch

from .base_tensor_conversion import BaseTensorConversion


class OperationTypes(Enum):
    ADDITION = 0
    SUBTRACTION = 1
    MULTIPLICATION = 2
    DIVISION = 3


class ArithmeticTensorConversion(BaseTensorConversion):
    def __init__(
        self,
        operation: OperationTypes,
        value: float | int | torch.Tensor,
        input_filter: Optional[Callable] = None,
        output_filter: Optional[Callable] = None,
    ):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.operation = operation
        self.value = value

    def handle_conversion(self, input_value, *full_context):
        match self.operation:
            case OperationTypes.ADDITION:
                return input_value + self.value
            case OperationTypes.SUBTRACTION:
                return input_value - self.value
            case OperationTypes.MULTIPLICATION:
                return input_value * self.value
            case OperationTypes.DIVISION:
                return input_value / self.value

    def __repr__(self):
        return f"Is the following arithmetic operation: {self.operation} and value: {self.value}"
