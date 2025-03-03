from .types import FIELD_SET
from .base_weight_conversion import BaseWeightConversion
from transformer_lens.weight_conversion.conversion_utils.weight_conversion_utils import WeightConversionUtils


class WeightConversionSet(BaseWeightConversion):
    def __init__(
        self,
        weights: FIELD_SET,
    ):
        super().__init__()
        self.weights = weights

    def handle_conversion(self, input_value):
        result = {}
        for weight_name in self.weights:
            result[weight_name] = self.process_conversion_action(
                input_value,
                conversion_details=self.weights[weight_name],
            )

        return result

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
