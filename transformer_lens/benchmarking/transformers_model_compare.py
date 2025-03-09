from .types import LAYER_RESULT

class TransformersModelCompare:
    
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        
    def compare(self)  -> LAYER_RESULT:
        return []