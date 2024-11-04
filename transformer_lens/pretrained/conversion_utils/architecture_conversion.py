from typing import List

from .conversion_steps.base_weight_conversion import BaseWeightConversion, FIELD_SET



class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        self.fields = fields
        
    def convert(self, remote_model):
        pass
    
    
    
