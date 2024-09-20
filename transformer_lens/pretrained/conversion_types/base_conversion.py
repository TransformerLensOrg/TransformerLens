from typing import List

class BaseConversion:
    def __init__(self, original_key: str) -> None:
        self.original_key = original_key
        self.key_levels = original_key.split(".")
        
    def convert(self, foreign_model):
        pass
    
    
    def find_weight(self, foreign_model):
        return self.search_next_level(self.key_levels, foreign_model)
    
    def search_next_level(self, levels: List[str], last_level):
        first_key = levels.pop(0)
        
        current_level = getattr(last_level, first_key)
        
        if len(levels) > 0:
            return self.search_next_level(levels, current_level)
        
        return current_level