"""Shared mock modules and fixtures for PretrainArchitectureAdapter and
PretrainModelContainer unit tests. Not a test file itself (no `test_`
prefix, so pytest won't collect it) -- imported by both
test_pretrain_adapter.py and test_pretrain_model_container.py.

Fully self-contained: builds tiny mock `nn.Module`s that reproduce the
relevant module tree (embed / blocks / norm1 / attn / norm2 / mlp / norm_f /
lm_head) rather than depending on any external training-framework package.
No weight loading, no network access.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.supported_architectures.pretrain import (
    ARCHITECTURE_NAME,
)

# --------------------------------------------------------------------------
# Tiny mock module tree, matching the real target architecture's
# block-calling convention (cos/sin as extra positional args -- see
# tiny_pretrain_model.py's module docstring). Real RoPE/RMSNorm math isn't
# needed here (covered by integration parity tests); only correct
# attribute names/shapes/call-signatures for component_mapping to wire up.
# --------------------------------------------------------------------------


class TinyRMSNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class TinyAttention(nn.Module):
    """Opaque attention stand-in: linear-in, linear-out, no real RoPE math.
    Takes (and ignores) cos/sin positionally, matching the real target
    architecture's block-calling convention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (d**0.5), dim=-1)
        return self.out(attn @ v)


class TinyDenseMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TinyMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([TinyDenseMLP(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x), dim=-1)
        topk_weights, topk_idx = weights.topk(self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for expert_idx in range(len(self.experts)):
            mask = (topk_idx == expert_idx).any(dim=-1, keepdim=True)
            if mask.any():
                out = out + mask * self.experts[expert_idx](x)
        return out


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, mlp: nn.Module):
        super().__init__()
        self.norm1 = TinyRMSNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.norm2 = TinyRMSNorm(d_model)
        self.mlp = mlp

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyPretrainModel(nn.Module):
    """Minimal decoder-only model matching the adapter's expected paths:
    embed / blocks / norm_f / lm_head. `moe_layer_indices` (possibly empty)
    controls which blocks get a `TinyMoE` mlp instead of a dense one.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        vocab_size: int = 256,
        n_experts: int = 4,
        top_k: int = 2,
        moe_layer_indices: frozenset[int] = frozenset(),
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TinyBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    mlp=(
                        TinyMoE(d_model, d_ff, n_experts, top_k)
                        if i in moe_layer_indices
                        else TinyDenseMLP(d_model, d_ff)
                    ),
                )
                for i in range(n_layers)
            ]
        )
        self.norm_f = TinyRMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.d_head = d_model // n_heads

    def _forward_impl(self, tokens: torch.Tensor) -> dict:
        b, s = tokens.shape
        x = self.embed(tokens)
        # Dummy cos/sin (no real RoPE math needed for these structural
        # tests) but real tensors, passed positionally into every block --
        # matches the real target architecture's convention.
        cos = torch.ones(s, self.d_head // 2)
        sin = torch.zeros(s, self.d_head // 2)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        return {"logits": self.lm_head(x)}

    def forward(self, tokens: torch.Tensor, **kwargs) -> dict:
        # **kwargs declared so pytest fixtures can pass through
        # run_with_cache's force-injected kwargs -- the actual kwarg-
        # forwarding contracts (strict vs. **kwargs-accepting source
        # forwards) are exercised directly against ForwardStrict/
        # ForwardVarKwargs below instead.
        return self._forward_impl(tokens)


class ForwardStrict(nn.Module):
    """Declares no **kwargs catch-all -- exercises the "must filter"
    branch of PretrainModelContainer's kwarg handling."""

    def __init__(self):
        super().__init__()
        self.seen_kwargs: dict = {}

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> dict:
        self.seen_kwargs = {"targets": targets}
        return {"logits": tokens.float()}


class ForwardVarKwargs(nn.Module):
    """Declares **kwargs -- exercises the "pass everything through" branch
    of PretrainModelContainer's kwarg handling."""

    def __init__(self):
        super().__init__()
        self.seen_kwargs: dict = {}

    def forward(self, tokens: torch.Tensor, **kwargs) -> dict:
        self.seen_kwargs = kwargs
        return {"logits": tokens.float()}


class MalformedMLP(nn.Module):
    """Has neither a dense MLP's (gate, up, down) nor an MoE layer's
    (router, experts) attributes -- used to test the adapter's failure
    path."""

    def __init__(self, d_model: int):
        super().__init__()
        self.some_other_layer = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.some_other_layer(x)


def make_cfg(d_model: int = 32, n_layers: int = 2) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // 4,
        n_layers=n_layers,
        n_ctx=128,
        n_heads=4,
        d_vocab=256,
        d_mlp=d_model * 2,
        architecture=ARCHITECTURE_NAME,
    )
