from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float

from transformer_lens.components import MLP, GatedMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class MoE(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)

        # Ensure that num_experts and experts_per_token are specified and non-zero
        assert self.cfg.num_experts is not None, "num_experts must be specified for MoE layer"
        assert self.cfg.experts_per_token, "experts_per_token must be specified for MoE layer"
        self.experts_per_token: int = self.cfg.experts_per_token
        assert (
            self.cfg.experts_per_token <= self.cfg.num_experts
        ), "experts_per_token must be less than or equal to num_experts"

        self.experts = nn.ModuleList(
            [
                GatedMLP(self.cfg) if self.cfg.gated_mlp else MLP(self.cfg)
                for _ in range(self.cfg.num_experts)
            ]
        )
        self.W_gate = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.num_experts, dtype=self.cfg.dtype)
        )

        # Hook on the weights of selected experts [batch pos experts_per_token]
        self.hook_expert_weights = HookPoint()
        # Hook on the indices of selected experts [batch pos experts_per_token]
        self.hook_expert_indices = HookPoint()

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # [batch, pos, d_model] -> [batch, pos, num_experts]
        gate_logits = einsum(
            "batch pos d_model, d_model num_experts -> batch pos num_experts",
            x,
            self.W_gate,
        )

        # choose the top k(=experts_per_token) experts to use
        # both are [batch, pos, experts_per_token]
        weights, expert_indices = torch.topk(gate_logits, self.experts_per_token)
        weights = self.hook_expert_weights(F.softmax(weights, dim=-1))
        expert_indices = self.hook_expert_indices(expert_indices)

        results = torch.zeros_like(x)
        for i, expert_mlp in enumerate(self.experts):
            # find the batch, pos, and expert indices which use this expert
            batch, pos, expert = torch.where(expert_indices == i)
            # accumulate the weighted outputs from the expert
            results[batch] += weights[batch, pos, expert, None, None] * expert_mlp(x[batch])

        return results
