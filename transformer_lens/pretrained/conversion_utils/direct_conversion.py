from .base_conversion import BaseConversion

class DirectConversion(BaseConversion):
    
    def convert(self, foreign_model):
        return self.find_weight(foreign_model)
        