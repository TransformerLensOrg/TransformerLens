from .base_weight_conversion import BaseWeightConversion


class CallableWeightConversion(BaseWeightConversion):
    def __init__(self, convert_callable: callable):
        self.convert_callable = convert_callable

    def convert(self, input_value: dict):
        return self.convert_callable(input_value)
