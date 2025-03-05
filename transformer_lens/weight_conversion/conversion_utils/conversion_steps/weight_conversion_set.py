import torch

from transformer_lens.weight_conversion.conversion_utils.conversion_helpers import (
    find_property,
)
from transformer_lens.weight_conversion.conversion_utils.weight_conversion_utils import (
    WeightConversionUtils,
)

from .base_weight_conversion import BaseWeightConversion
from .types import CONVERSION_ACTION, FIELD_SET


class WeightConversionSet(BaseWeightConversion):
    def __init__(
        self,
        weights: FIELD_SET,
    ):
        super().__init__()
        self.weights = weights

    def handle_conversion(self, input_value, *full_context):
        result = {}
        for weight_name in self.weights:
            conversion_action = self.weights[weight_name]
            result[weight_name] = self.process_conversion_action(
                input_value,
                conversion_details=conversion_action,
            )

        return result

    def process_conversion_action(
        self, input_value, conversion_details: CONVERSION_ACTION, *full_context
    ):
        if isinstance(conversion_details, torch.Tensor):
            return conversion_details
        elif isinstance(conversion_details, str):
            return find_property(conversion_details, input_value)
        else:
            (remote_field, conversion) = conversion_details
            return self.process_conversion(input_value, remote_field, conversion, *full_context)

    def process_conversion(
        self, input_value, remote_field: str, conversion: BaseWeightConversion, *full_context
    ):
        field = find_property(remote_field, input_value)
        if isinstance(field, WeightConversionSet):
            result = []
            for layer in field:
                result.append(conversion.convert(layer, input_value, *full_context))
            return result

        else:
            return conversion.convert(field, *[input_value, *full_context])

    def __repr__(self):
        conversion_string = (
            "Is composed of a set of nested conversions with the following details {\n\t"
        )
        # This is a bit of a hack to get the string representation of nested conversions
        conversion_string += WeightConversionUtils.create_conversion_string(self.weights)[
            :-1
        ].replace("\n", "\n\t")
        conversion_string += "\n}"
        return conversion_string
