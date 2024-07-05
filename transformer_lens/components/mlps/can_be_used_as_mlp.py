"""Can Be Used as MLP component.

This module serves as the base for everything within TransformerLens that can be used like an MLP.
This does not necessarily mean that every component extending this class will be an MLP, but 
everything extending this class can be used interchangeably for an MLP.
"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.factories.activation_function_factory import (
    ActivationFunctionFactory,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.activation_functions import ActivationFunction


class CanBeUsedAsMLP(nn.Module):
    # The actual activation function
    act_fn: ActivationFunction

    # The full config object for the model
    cfg: HookedTransformerConfig

    # The d mlp value pulled out of the config to make sure it always has a value
    d_mlp: int

    # The middle hook point will be None unless it specifically should be used
    hook_mid: Optional[HookPoint]  # [batch, pos, d_mlp]

    # The layer norm component if the activation function is a layer norm
    ln: Optional[nn.Module]

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        """The base init for all MLP like components

        Args:
            config (Union[Dict, HookedTransformerConfig]): The config for this instance

        Raises:
            ValueError: If there is a misconfiguration
        """
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        if self.cfg.d_mlp is None:
            raise ValueError("d_mlp must be set to use an MLP")

        self.d_mlp = self.cfg.d_mlp

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """The format for all forward functions for any MLP"""
        return x

    def select_activation_function(self) -> None:
        """This function should be called by all components in their init to get everything needed
        for activation functions setup.

        Raises:
            ValueError: If the configure activation function is not supported.
        """

        self.act_fn = ActivationFunctionFactory.pick_activation_function(self.cfg)

        if self.cfg.is_layer_norm_activation():
            self.hook_mid = HookPoint()
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)
