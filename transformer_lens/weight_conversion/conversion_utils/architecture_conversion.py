from typing import List

from .conversion_steps.base_weight_conversion import BaseWeightConversion, FIELD_SET
from .conversion_steps.weight_conversion_set import WeightConversionSet

class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        self.field_set = WeightConversionSet(fields)
        
    def convert(self, remote_model):
        return self.field_set.convert(input_value=remote_model)
    

