import torch


from transformer_lens.weight_conversion.conversion_utils.model_search import (
    find_property,
)

CONVERSION = tuple[str, "BaseWeightConversion"]
CONVERSION_ACTION = torch.Tensor | str | CONVERSION
FIELD_SET = dict[str, torch.Tensor | str | CONVERSION]

class BaseWeightConversion:
    
    def __init__(self, input_filter: callable|None = None, output_filter: callable|None = None):
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
            return conversion.convert(weight)

    def handle_conversion(self, input_value):
        raise Exception(
            f"The conversion function for {type(self).__name__} needs to be implemented."
        )

