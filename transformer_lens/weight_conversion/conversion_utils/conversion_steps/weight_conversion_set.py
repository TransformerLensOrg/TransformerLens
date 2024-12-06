import torch
from transformer_lens.weight_conversion.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion, CONVERSION, FIELD_SET

class WeightConversionSet(BaseWeightConversion):
    
    def __init__(self, weights: FIELD_SET):
        self.weights = weights
    
    def convert(self, input_value):
        result = {}
        for weight_name in self.weights:
            result[weight_name] = self.process_weight_conversion(
                input_value,
                conversion_details=self.weights[weight_name],
            )
                
            
        return result
    
    def process_weight_conversion(self, input_value, conversion_details: torch.Tensor|CONVERSION):
        if isinstance(conversion_details,  torch.Tensor):
            return conversion_details
        else: 
            (remote_field, conversion) = conversion_details
            weight = find_property(remote_field, input_value)
            return conversion.convert(weight)