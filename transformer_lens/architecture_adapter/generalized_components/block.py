import torch
import torch.nn as nn

from .base import GeneralizedComponent


class BlockBridge(GeneralizedComponent):
    def __init__(self, original_component):
        super().__init__(original_component, "block")
        self.original_component = original_component
        # Proxy submodules for both TL and HF naming
        self.ln1 = getattr(original_component, 'ln1', None)
        self.attn = getattr(original_component, 'attn', None)
        self.ln2 = getattr(original_component, 'ln2', None)
        self.mlp = getattr(original_component, 'mlp', None)
        self.self_attn = getattr(original_component, 'self_attn', self.attn)
        # Add more aliases if needed

    def forward(self, *args, **kwargs):
        return self.original_component(*args, **kwargs) 