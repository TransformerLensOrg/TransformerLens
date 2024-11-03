from typing import List

from .base_field_conversion import FIELD_SET



class ArchitectureConversion:
    def __init__(self, fields: FIELD_SET) -> None:
        # TODO change to take in list of base field conversions
        self.fields = fields
        
    def convert(self, foreign_model):
        pass
    
    
    
