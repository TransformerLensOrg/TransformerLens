from dataclasses import dataclass

import torch as t
import torch.nn as nn
from torchtyping import TensorType as TT

from . import encoder_layer
from .config import Config


@dataclass
class Output:
    hidden_states: TT[
        "n_layers", "batch", "seq", "hidden"
    ]  # slightly different than HF which stacks embeddings and hidden states
    attentions_post_softmax: TT[
        "n_layers", "batch", "head", "seq", "seq"
    ]  # similar to HF:  https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM.forward.output_attentions


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [encoder_layer.EncoderLayer(config) for _ in range(config.layers)]
        )

    def forward(self, x, mask=None) -> Output:
        # TODO maybe make this a list so that some grad fns are simpler
        hidden_states = []
        attentions_post_softmax = []
        for layer in self.layers:
            # TODO does this kill performance? vs. nn.sequential (and with using a list)
            input_ = x if len(hidden_states) == 0 else hidden_states[-1]
            layer_output: encoder_layer.Output = layer(input_, mask)
            hidden_states.append(layer_output.hidden_state)
            attentions_post_softmax.append(layer_output.attention_post_softmax)
        return Output(
            hidden_states=t.stack(hidden_states),
            attentions_post_softmax=t.stack(attentions_post_softmax),
        )
