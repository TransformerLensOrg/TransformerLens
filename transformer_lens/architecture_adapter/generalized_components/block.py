import torch.nn as nn

from .base import GeneralizedComponent

class BlockBridge(GeneralizedComponent):
    def __init__(self, original_component: nn.Module, name: str):
        super().__init__(original_component, name)

    def forward(self, *args, **kwargs):
        return self.original_component(*args, **kwargs) 