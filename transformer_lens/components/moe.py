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
        assert (
            self.cfg.num_experts is not None
        ), "num_experts must be specified for MoE layer"
        assert (
            self.cfg.experts_per_token
        ), "experts_per_token must be specified for MoE layer"
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
        router_logits = F.linear(x, self.W_gate.T)
        router_logits = router_logits.view(
            x.shape[0] * x.shape[1], self.cfg.num_experts
        )
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        routing_weights, expert_indices = torch.topk(
            routing_weights, self.experts_per_token
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        routing_weights = self.hook_expert_weights(
            routing_weights
        )  # F.softmax(weights, dim=-1))
        expert_indices = self.hook_expert_indices(expert_indices)
        expert_mask = F.one_hot(
            expert_indices, num_classes=self.cfg.num_experts
        ).permute(2, 1, 0)

        results = torch.zeros(
            x.shape[0] * x.shape[1], self.cfg.d_model, device=x.device
        )
        hidden_states = x.view(-1, self.cfg.d_model)
        for i, expert_mlp in enumerate(self.experts):
            # find the batch, pos, and expert indices which use this expert
            idx, top_x = torch.where(expert_mask[i])

            current_state = hidden_states[None, top_x].reshape(-1, self.cfg.d_model)

            expert_hidden_state = expert_mlp.act_fn(
                F.linear(current_state, expert_mlp.W_gate.T)
            ) * F.linear(current_state, expert_mlp.W_in.T)
            expert_output = F.linear(expert_hidden_state, expert_mlp.W_out.T)

            current_hidden_states = expert_output * routing_weights[top_x, idx, None]

            results.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        return results.reshape_as(x)
