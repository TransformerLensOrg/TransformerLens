import torch as t
import torch.nn as nn
from torchtyping import TensorType as TT

from . import encoder_layer
from .config import Config


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [encoder_layer.EncoderLayer(config) for _ in range(config.layers)]
        )

    def forward(
        self, x: TT["batch", "seq", "hidden"], mask=None
    ) -> TT["batch", "seq", "hidden"]:
        # TODO someday write a demo with hooks that show how to gather hidden states
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
