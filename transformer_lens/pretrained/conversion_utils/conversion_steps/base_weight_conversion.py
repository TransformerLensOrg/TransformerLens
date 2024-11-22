class BaseWeightConversion:
    
    def __init__(self, original_key: str):
        self.original_key = original_key
        
        
    def convert(self, remote_weights: dict):
        raise Exception(f"The conversion function for {type(self).__name__} needs to be implemented.")
    

FIELD_SET = dict[str, BaseWeightConversion]