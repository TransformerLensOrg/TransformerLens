class BaseFieldConversion:
    
    def __init__(self, original_key: str):
        self.original_key = original_key
        self.key_levels = original_key.split(".")

    def find_weight(self, foreign_model):
        return self.search_next_level(self.key_levels, foreign_model)