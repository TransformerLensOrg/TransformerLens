from transformer_lens.pretrained.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion

class DirectWeightConversion(BaseWeightConversion):
    
    def convert(self, remote_weights: dict):
        return find_property(self.original_key, remote_weights)