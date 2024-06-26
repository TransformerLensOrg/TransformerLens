from typing import Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import gelu_fast, gelu_new, solu

ActivationFunction = Callable[..., torch.Tensor]
SUPPORTED_ACTIVATION_FUNCTIONS: dict[str, ActivationFunction] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "solu_ln": solu,
}

class CanBeUsedAsMLP(nn.Module):
    
    act_fn: ActivationFunction
    hook_mid: Optional[HookPoint] = None
    
    ln: Optional[nn.Module] = None
    
    def __init__(self, config: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(config)
        if self.cfg.d_mlp is None:
            raise ValueError("d_mlp must be set to use an MLP")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        pass
        
    def select_activation_function(self):
        
        activation_function = SUPPORTED_ACTIVATION_FUNCTIONS.get(self.cfg.act_fn)
            
        if (activation_function is None):
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")
        
        self.act_fn = activation_function
        
        if self.cfg.act_fn == "solu_ln":
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)