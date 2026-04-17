from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class ClassifierHead(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)

        self.num_labels = self.cfg.num_labels

        if self.num_labels > 0:
            self.W: Float[torch.Tensor, "d_model num_labels"] = nn.Parameter(
                torch.empty(self.cfg.d_model, self.num_labels, dtype=self.cfg.dtype)
            )
            self.b: Float[torch.Tensor, "num_labels"] = nn.Parameter(
                torch.zeros(self.num_labels, dtype=self.cfg.dtype)
            )
        else:
            self.W = None
            self.b = None

        # Hooks (mirrors Unembed)
        self.hook_in = HookPoint()   # [batch, d_model]
        self.hook_out = HookPoint()  # [batch, num_labels]

    def forward(
        self,
        residual: Float[torch.Tensor, "batch d_model"],
    ) -> Float[torch.Tensor, "batch num_labels"]:

        residual = self.hook_in(residual)

        if self.num_labels <= 0:
            return residual

        # Match HF nn.Linear exactly (important for weight porting)
        logits = F.linear(residual, self.W.T.contiguous(), self.b)

        return self.hook_out(logits)
