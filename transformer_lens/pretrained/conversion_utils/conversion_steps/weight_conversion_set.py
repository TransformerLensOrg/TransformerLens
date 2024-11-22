from transformer_lens.pretrained.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion, FIELD_SET

class WeightConversionSet(BaseWeightConversion):
    
    def __init__(self, original_key: str, fields: FIELD_SET):
        super().__init__(original_key)
        self.fields = fields
    
    def convert(self, remote_weights: dict):
        modules = find_property(self.original_key, remote_weights)
        result = []
        for module in modules:
            field_set = {}
            for field_name in self.fields:
                field = self.fields[field_name]
                field_set[field_name] = field.convert(module)
                
            result.append(field_set)
            
        return result