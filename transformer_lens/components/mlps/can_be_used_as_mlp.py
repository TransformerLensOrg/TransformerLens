"""Can Be Used as MLP component.

This module serves as the base for everything within TransformerLens that can be used like an MLP.
This does not necessarily mean that every component extending this class will be an MLP, but 
everything extending this class can be used interchangeably for an MLP.
"""
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import gelu_fast, gelu_new, solu

# Convenient type for the format of each activation function
ActivationFunction = Callable[..., torch.Tensor]
# All currently supported activation functions. To add a new function, simply
# put the name of the function as the key, and the value as the actual callable.
SUPPORTED_ACTIVATION_FUNCTIONS: dict[str, ActivationFunction] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "solu_ln": solu,
}

class CanBeUsedAsMLP(nn.Module):
    
    # The actual activation function
    act_fn: ActivationFunction
    
    # The middle hook point will be None unless it specifically should be used
    hook_mid: Optional[HookPoint] # [batch, pos, d_mlp]
    
    # The layer norm component if the activation function is a layer norm
    ln: Optional[nn.Module]
    
    def __init__(self, config: Union[Dict, HookedTransformerConfig]):
        """The base init for all MLP like components

        Args:
            config (Union[Dict, HookedTransformerConfig]): The config for this instance

        Raises:
            ValueError: If there is a misconfiguration
        """
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(config)
        if self.cfg.d_mlp is None:
            raise ValueError("d_mlp must be set to use an MLP")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """The format for all forward functions for any MLP
        """
        pass
        
    def select_activation_function(self):
        """This function should be called by all components in their init to get everything needed
        for activation functions setup.

        Raises:
            ValueError: If the configure activation function is not supported.
        """
        
        activation_function = SUPPORTED_ACTIVATION_FUNCTIONS.get(self.cfg.act_fn)
            
        if (activation_function is None):
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")
        
        self.act_fn = activation_function
        
        if self.cfg.is_layer_norm_activation():
            self.hook_mid = HookPoint()
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)