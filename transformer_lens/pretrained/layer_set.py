from typing import Dict
from .weight_conversion import WeightConversion

class LayerSet(Dict):
    def __init__(self, conversion_instructions, dict):
        self.weight_conversions = {transformer_lens_key:
            WeightConversion(conversion_instruction) for transformer_lens_key, conversion_instruction in conversion_instructions.items()
        }