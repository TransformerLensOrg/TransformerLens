from typing import List


class WeightConversion:
    
    def __init__(self, path: str):
        self.path = path
        self.parts = path.split(".")
    
    def convert(self, transformers_model):
        return self.select_weight(transformers_model)
    
    def select_weight(self, transformers_model):
        
        return self.select_next_level(self.parts, transformers_model)
    
    def select_next_level(self, levels: List, previous_level):
        first_key = levels.pop(0)
        
        current_level = getattr(previous_level, first_key)
        
        if len(levels) > 0:
            return self.select_next_level(levels, current_level)
        
        return current_level
    
    
    
    def revert():
        pass
    