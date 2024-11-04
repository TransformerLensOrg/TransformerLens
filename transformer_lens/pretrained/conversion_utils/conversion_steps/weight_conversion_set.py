from transformer_lens.pretrained.conversion_utils.model_search import find_weight
from .base_weight_conversion import BaseWeightConversion

class WeightConversionSet(BaseWeightConversion):
    
    def convert(self, remote_model):
        return find_weight(self.original_key, remote_model)