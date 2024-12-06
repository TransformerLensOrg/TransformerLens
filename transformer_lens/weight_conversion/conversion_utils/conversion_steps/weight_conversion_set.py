from transformer_lens.weight_conversion.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion, FIELD_SET

class WeightConversionSet(BaseWeightConversion):
    
    def __init__(self, original_key: str, weights: FIELD_SET):
        super().__init__(original_key)
        self.weights = weights
    
    def convert(self, remote_weights):
        weights = find_property(self.original_key, remote_weights) if self.original_key else remote_weights
        result = {}
        for weight_name in self.weights:
            field = self.weights[weight_name]
            result[weight_name] = field.convert(weights)
                
            
        return result