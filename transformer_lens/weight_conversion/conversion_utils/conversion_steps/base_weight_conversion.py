from collections.abc import Callable
from typing import Optional
from transformer_lens.weight_conversion.conversion_utils.conversion_helpers import find_property
from .types import WeightConversionInterface, CONVERSION


class BaseWeightConversion(WeightConversionInterface):
    def __init__(
        self, input_filter: Optional[Callable] = None, output_filter: Optional[Callable] = None
    ):
        self.input_filter = input_filter
        self.output_filter = output_filter

    def convert(self, input_value, *full_context):
        input_value = (
            self.input_filter(input_value) if self.input_filter is not None else input_value
        )
        output = self.handle_conversion(input_value, *full_context)
        return self.output_filter(output) if self.output_filter is not None else output


    def handle_conversion(self, input_value, *full_context):
        raise NotImplementedError(
            f"The conversion function for {type(self).__name__} needs to be implemented."
        )
            
    def process_conversion(self, input_value, remote_field: str, conversion: CONVERSION, *full_context):
        field = find_property(remote_field, input_value)
        if isinstance(field, WeightConversionSet):
            result = []
            for layer in field:
                result.append(conversion.convert(layer, input_value, *full_context))
            return result

        else:
            return conversion.convert(field, input_value, *full_context)
