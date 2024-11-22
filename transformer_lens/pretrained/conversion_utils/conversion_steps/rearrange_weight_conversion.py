import einops

from transformer_lens.pretrained.conversion_utils.model_search import find_property
from .base_weight_conversion import BaseWeightConversion

class RearrangeWeightConversion(BaseWeightConversion):

    def __init__(self, original_key: str, pattern: str, **axes_lengths):
        super().__init__(original_key)
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        
    def convert(self, remote_weights: dict):
        field = find_property(self.original_key, remote_weights)
        return einops.rearrange(field, self.pattern, **self.axes_lengths)