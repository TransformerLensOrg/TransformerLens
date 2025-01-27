import torch
from typing import TypeAlias, Optional, Union


from transformer_lens.weight_conversion.conversion_utils.model_search import (
    find_property,
)

CONVERSION: TypeAlias = tuple[str, "BaseWeightConversion"]
CONVERSION_ACTION: TypeAlias = Union[torch.Tensor | str | CONVERSION]
FIELD_SET: TypeAlias = dict[str, Union[torch.Tensor | str | CONVERSION]]

class BaseWeightConversion:
    
    def __init__(self, input_filter: Optional[callable] = None, output_filter: Optional[callable] = None):
        self.input_filter = input_filter
        self.output_filter = output_filter
        
    def convert(self, input_value):
        input_value = self.input_filter(input_value) if self.input_filter is not None else input_value
        output = self.handle_conversion(input_value)
        return self.output_filter(output) if self.output_filter is not None else output
    

    def process_weight_conversion(self, input_value, conversion_details: CONVERSION_ACTION):
        if isinstance(conversion_details, torch.Tensor):
            return conversion_details
        elif isinstance(conversion_details, str):
            return find_property(conversion_details, input_value)
        else:
            (remote_field, conversion) = conversion_details
            weight = find_property(remote_field, input_value)
            if isinstance(conversion, "WeightConversionSet"):
                result = []
                for layer in weight:
                    result.append(conversion.convert(layer))
                return result
                    
            else:
                return conversion.convert(weight)

    def handle_conversion(self, input_value):
        raise Exception(
            f"The conversion function for {type(self).__name__} needs to be implemented."
        )

