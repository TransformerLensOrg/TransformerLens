from collections.abc import Callable
from typing import Optional


class BaseWeightConversion:
    def __init__(
        self, input_filter: Optional[Callable] = None, output_filter: Optional[Callable] = None
    ):
        self.input_filter = input_filter
        self.output_filter = output_filter

    def convert(self, input_value):
        input_value = (
            self.input_filter(input_value) if self.input_filter is not None else input_value
        )
        output = self.handle_conversion(input_value)
        return self.output_filter(output) if self.output_filter is not None else output


    def handle_conversion(self, input_value):
        raise NotImplementedError(
            f"The conversion function for {type(self).__name__} needs to be implemented."
        )
