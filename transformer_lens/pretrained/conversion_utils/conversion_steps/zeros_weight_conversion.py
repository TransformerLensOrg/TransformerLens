import torch
from .base_weight_conversion import BaseWeightConversion

class ZerosWeightConversion(BaseWeightConversion):
    
    def __init__(self, *size: int):
        self.size = size
    
    def convert(self, remote_field):
        return torch.zeros(self.size)