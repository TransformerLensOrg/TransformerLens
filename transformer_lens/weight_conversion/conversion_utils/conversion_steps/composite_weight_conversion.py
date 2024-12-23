from .base_weight_conversion import BaseWeightConversion


class CompositeWeightConversion(BaseWeightConversion):
    def __init__(self, steps: list[BaseWeightConversion]):
        self.steps = steps

    def convert(self, input_value):
        result = input_value
        for step in self.steps:
            result = step.convert(result)

        return result
