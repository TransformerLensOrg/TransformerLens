from .base_weight_conversion import BaseWeightConversion

class DirectWeightConversion(BaseWeightConversion):
    
    def convert(self, remote_field):
        return remote_field