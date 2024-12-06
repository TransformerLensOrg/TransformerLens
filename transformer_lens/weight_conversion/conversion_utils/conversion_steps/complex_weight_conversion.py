from transformer_lens.weight_conversion.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion

class ComplexWeightConversion(BaseWeightConversion):
    
    def __init__(self, original_key: str, convert_callable: callable):
        super().__init__(original_key)
        self.convert_callable = convert_callable
    
    def convert(self, remote_weights: dict):
        field = find_property(self.original_key, remote_weights)
        
        return self.convert_callable(field)