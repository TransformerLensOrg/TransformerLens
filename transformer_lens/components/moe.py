from typing import Dict, Union

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float

from transformer_lens.components import MLP, GatedMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.d_mlp
        self.hidden_dim = config.d_model

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MoE(nn.Module):
    def __init__(self, config: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(config)

        # Ensure that num_experts and experts_per_token are specified and non-zero
        assert self.cfg.num_experts is not None, "num_experts must be specified for MoE layer"
        assert self.cfg.experts_per_token, "experts_per_token must be specified for MoE layer"
        
        self.num_experts: int = self.cfg.num_experts
        self.experts_per_token: int = self.cfg.experts_per_token

        assert (
            self.cfg.experts_per_token <= self.cfg.num_experts
        ), "experts_per_token must be less than or equal to num_experts"

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(self.cfg) for _ in range(self.num_experts)])
        self.W_gate = nn.Linear(self.cfg.d_model, self.cfg.num_experts, bias=False)

        # Hook on the weights of selected experts [batch pos experts_per_token]
        self.hook_expert_weights = HookPoint()
        # Hook on the indices of selected experts [batch pos experts_per_token]
        self.hook_expert_indices = HookPoint()

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # [batch, pos, d_model] -> [batch, pos, num_experts]
        batch, pos, d_model = x.shape
        x = x.view(-1, d_model)
        gate_logits = self.W_gate(x)

        # choose the top k(=experts_per_token) experts to use
        # both are [batch, pos, experts_per_token]
        weights = self.hook_expert_weights(F.softmax(gate_logits, dim=1, dtype=torch.float))
        weights, expert_indices = torch.topk(weights, self.experts_per_token, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        expert_indices = self.hook_expert_indices(expert_indices)
        weights = weights.to(gate_logits.dtype)

        results = torch.zeros(
            (batch * pos, d_model), dtype=x.dtype, device=x.device
        )
        expert_mask = torch.nn.functional.one_hot(expert_indices, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, d_model)
            current_hidden_states = expert_layer(current_state) * x[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            results.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        results = results.reshape(batch, pos, d_model)
        return results
