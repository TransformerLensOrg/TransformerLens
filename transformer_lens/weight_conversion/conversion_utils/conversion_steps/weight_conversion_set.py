from .base_weight_conversion import FIELD_SET, BaseWeightConversion


class WeightConversionSet(BaseWeightConversion):
    def __init__(self, weights: FIELD_SET, input_filter: callable|None = None, output_filter: callable|None = None):
        super().__init__(input_filter=input_filter, output_filter=output_filter)
        self.weights = weights

    def handle_conversion(self, input_value):
        result = {}
        for weight_name in self.weights:
            result[weight_name] = super().process_weight_conversion(
                input_value,
                conversion_details=self.weights[weight_name],
            )

        return result
