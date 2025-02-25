from collections.abc import Callable

from .base_weight_conversion import BaseWeightConversion


class CallableWeightConversion(BaseWeightConversion):
    def __init__(self, convert_callable: Callable):
        super().__init__()
        self.convert_callable = convert_callable

    def handle_conversion(self, input_value: dict):
        return self.convert_callable(input_value)

    def __repr__(self):
        return f"Is the following callable operation: {self.convert_callable}"
