from typing import Callable

from .base_weight_conversion import BaseWeightConversion


class CompositeWeightConversion(BaseWeightConversion):
    def __init__(
        self,
        steps: list[BaseWeightConversion],
        input_filter: Callable | None = None,
        output_filter: Callable | None = None,
    ):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.steps = steps

    def handle_conversion(self, input_value):
        result = input_value
        for step in self.steps:
            result = step.convert(result)

        return result

    def __repr__(self):
        return f"Is a composite operation with the following steps: {self.steps}"
