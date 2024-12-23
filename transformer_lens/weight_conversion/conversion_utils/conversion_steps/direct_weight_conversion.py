from .base_weight_conversion import BaseWeightConversion


class DirectWeightConversion(BaseWeightConversion):
    def convert(self, input_value):
        return input_value
