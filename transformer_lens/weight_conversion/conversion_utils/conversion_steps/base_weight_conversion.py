from collections.abc import Callable
import torch
from typing import Optional
from transformer_lens.weight_conversion.conversion_utils.conversion_helpers import find_property
from .types import WeightConversionInterface, CONVERSION_ACTION, CONVERSION


class BaseWeightConversion(WeightConversionInterface):
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
        

    def process_conversion_action(self, input_value, conversion_details: CONVERSION_ACTION):
        if isinstance(conversion_details, torch.Tensor):
            return conversion_details
        elif isinstance(conversion_details, str):
            return find_property(conversion_details, input_value)
        else:
            (remote_field, conversion) = conversion_details
            return self.process_conversion(input_value, remote_field, conversion)
            
    def process_conversion(self, input_value, remote_field: str, conversion: CONVERSION):
        field = find_property(remote_field, input_value)
        if isinstance(field, WeightConversionSet):
            result = []
            for layer in field:
                result.append(conversion.convert(layer))
            return result

        else:
            return conversion.convert(field)
