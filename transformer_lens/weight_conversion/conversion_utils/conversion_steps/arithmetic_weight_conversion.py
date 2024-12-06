from enum import Enum
from transformer_lens.weight_conversion.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion

class OperationTypes(Enum):
    ADDITION = 0
    SUBTRACTION = 1
    MULTIPLICATION = 2
    DIVISION = 3

class ArithmeticWeightConversion(BaseWeightConversion):
    
    def __init__(self, original_key: str, operation: OperationTypes, value: float|int):
        super().__init__(original_key)
        self.operation = operation
        self.value = value
    
    def convert(self, remote_weights: dict):
        field = find_property(self.original_key, remote_weights)
        match self.operation:
            case OperationTypes.ADDITION:
                return field + self.value
            case OperationTypes.SUBTRACTION:
                return field - self.value
            case OperationTypes.MULTIPLICATION:
                return field * self.value
            case OperationTypes.DIVISION:
                return field / self.value