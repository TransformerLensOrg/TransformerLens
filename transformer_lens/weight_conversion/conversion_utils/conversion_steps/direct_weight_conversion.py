from transformer_lens.weight_conversion.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion

class DirectWeightConversion(BaseWeightConversion):
    
    def convert(self, remote_weights):
        return find_property(self.original_key, remote_weights)