from transformers.modelling_utils import PretrainedModel 
import torch
from typing import Dict
from .weight_conversion import WeightConversion



class BaseWithConversion:
    
    def __init__(self, root_layers: LayerSet):
        self.root_layers = root_layers
    
    def convert(self, transformers_model):
        state_dict = {}
        
        for key, weight_conversion in self.root_layers:
            state_dict[key] = weight_conversion.convert(transformers_model) 

        return state_dict
    
    def convert_layer_set(self, layer_set: LayerSet, transformers_model):
        
    
    def select_transformers_weight(self, transformer_lens_key: str, transformers_model: PreTrainedModel) -> torch.Tensor
        pass