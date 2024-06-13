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
        self.gate = nn.Linear(self.cfg.d_model, self.cfg.num_experts, bias=False)
        
    def calculate_expert_weights(
        self,
        i: int,
        x: Float[torch.Tensor, "batch pos d_model"],
        expert_indices: Float[torch.Tensor, "batch num_experts"],
        weights: Float[torch.Tensor, "batch num_experts"],
        expert_mlp: Union[GatedMLP, MLP],
    ) -> Float[torch.Tensor, "pos d_model"]:
        return weights[batch, pos, expert, None, None] * expert_mlp(x[batch])

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # [batch, pos, d_model] -> [batch, pos, num_experts]

        batch_size, sequence_length, hidden_dim = x.shape
        hidden_states = x.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.experts_per_token, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.cfg.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.cfg.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

