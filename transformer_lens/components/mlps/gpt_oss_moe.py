"""GPT-OSS Mixture of Experts implementation for TransformerLens.

GPT-OSS uses a unique MoE architecture:
- Merged expert weights (gate_up_proj with interleaved gate/up columns)
- Custom GLU activation: gate * sigmoid(gate * 1.702) * (up + 1), with clamping
- Router with bias, softmax applied AFTER top-k selection
- Expert projections have biases
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

GPT_OSS_ALPHA = 1.702
GPT_OSS_LIMIT = 7.0


class GptOssExpert(nn.Module):
    """Single GPT-OSS expert with custom GLU activation.

    The activation differs from standard SiLU:
        gate = clamp(x @ W_gate + b_gate, max=7.0)
        up   = clamp(x @ W_in + b_in, min=-7.0, max=7.0)
        glu  = gate * sigmoid(gate * 1.702)
        out  = (up + 1) * glu
        result = out @ W_out + b_out
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_mlp is not None

        self.W_gate = nn.Linear(cfg.d_model, cfg.d_mlp, bias=True, dtype=cfg.dtype)
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp, bias=True, dtype=cfg.dtype)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model, bias=True, dtype=cfg.dtype)

        self.hook_gate = HookPoint()
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: Float[torch.Tensor, "pos d_model"]) -> Float[torch.Tensor, "pos d_model"]:
        gate = self.hook_gate(self.W_gate(x))
        up = self.hook_pre(self.W_in(x))

        # GPT-OSS custom activation
        gate = gate.clamp(max=GPT_OSS_LIMIT)
        up = up.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
        glu = gate * torch.sigmoid(gate * GPT_OSS_ALPHA)
        post = self.hook_post((up + 1) * glu)

        return self.W_out(post)


class GptOssMoE(CanBeUsedAsMLP):
    """GPT-OSS Mixture of Experts layer.

    Differences from standard TransformerLens MoE (Mixtral):
    - Router has bias
    - Softmax applied AFTER top-k selection (not before)
    - Experts use custom GLU activation (not SiLU)
    - Expert projections have biases
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)

        assert self.cfg.num_experts is not None
        assert self.cfg.experts_per_token is not None

        self.num_experts: int = self.cfg.num_experts
        self.experts_per_token: int = self.cfg.experts_per_token

        self.experts = nn.ModuleList([GptOssExpert(self.cfg) for _ in range(self.num_experts)])
        # GPT-OSS router has bias (unlike Mixtral)
        self.W_gate = nn.Linear(
            self.cfg.d_model, self.cfg.num_experts, bias=True, dtype=self.cfg.dtype
        )

        self.hook_expert_weights = HookPoint()
        self.hook_expert_indices = HookPoint()

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        batch, pos, d_model = x.shape
        x = x.view(-1, d_model)

        # GPT-OSS routing: softmax AFTER top-k (differs from Mixtral)
        gate_logits = self.W_gate(x)
        top_values, expert_indices = torch.topk(gate_logits, self.experts_per_token, dim=-1)
        # Softmax over just the selected experts
        top_weights = F.softmax(top_values, dim=-1, dtype=torch.float)

        # Build full routing weights tensor for hooks (num_tokens, num_experts)
        routing_weights = torch.zeros_like(gate_logits, dtype=torch.float)
        routing_weights.scatter_(1, expert_indices, top_weights)

        routing_weights = self.hook_expert_weights(routing_weights)
        expert_indices = self.hook_expert_indices(expert_indices)
        routing_weights = routing_weights.to(x.dtype)

        results = torch.zeros((batch * pos, d_model), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            current_state = x[top_x]
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, expert_idx, None]
            )
            results.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        return results.reshape(batch, pos, d_model)
