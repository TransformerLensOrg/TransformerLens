import torch

from transformer_lens.weight_conversion.conversion_utils.model_search import (
    find_property,
)

from .base_weight_conversion import CONVERSION, FIELD_SET, BaseWeightConversion


class WeightConversionSet(BaseWeightConversion):
    def __init__(self, weights: FIELD_SET, input_filter: callable|None = None, output_filter: callable|None = None):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.weights = weights

    def handle_conversion(self, input_value):
        result = {}
        for weight_name in self.weights:
            result[weight_name] = self.process_weight_conversion(
                input_value,
                conversion_details=self.weights[weight_name],
            )

        return result

    def process_weight_conversion(self, input_value, conversion_details: torch.Tensor | str | CONVERSION):
        if isinstance(conversion_details, torch.Tensor):
            return conversion_details
        elif isinstance(conversion_details, str):
            return find_property(conversion_details, input_value)
        else:
            (remote_field, conversion) = conversion_details
            weight = find_property(remote_field, input_value)
            return conversion.convert(weight)
