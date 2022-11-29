import torch as t
import torch.nn as nn
from torchtyping import TensorType as TT

from .EasyBERTConfig import EasyBERTConfig
from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, x, mask=None) -> TT["n_layers", "batch", "seq", "hidden"]:
        # TODO document that this returns all layers
        # TODO maybe make this a list so that some grad fns are simpler
        intermediate = []
        for layer in self.layers:
            # TODO does this kill performance? vs. nn.sequential (and with using a list)
            input_ = x if len(intermediate) == 0 else intermediate[-1]
            intermediate.append(layer(input_, mask=mask))
        return t.stack(intermediate)
