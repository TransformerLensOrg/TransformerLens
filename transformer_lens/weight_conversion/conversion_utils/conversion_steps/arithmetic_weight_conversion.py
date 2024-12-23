from enum import Enum

import torch

from .base_weight_conversion import BaseWeightConversion


class OperationTypes(Enum):
    ADDITION = 0
    SUBTRACTION = 1
    MULTIPLICATION = 2
    DIVISION = 3


class ArithmeticWeightConversion(BaseWeightConversion):
    def __init__(self, operation: OperationTypes, value: float | int | torch.Tensor):
        self.operation = operation
        self.value = value

    def convert(self, input_value):
        match self.operation:
            case OperationTypes.ADDITION:
                return input_value + self.value
            case OperationTypes.SUBTRACTION:
                return input_value - self.value
            case OperationTypes.MULTIPLICATION:
                return input_value * self.value
            case OperationTypes.DIVISION:
                return input_value / self.value
