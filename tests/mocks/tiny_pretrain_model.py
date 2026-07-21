"""A tiny, self-contained decoder-only reference model (RoPE, RMSNorm,
gated SwiGLU MLP, optional MoE) used only by the pretrain-adapter
integration tests, to exercise real numerical parity rather than a purely
structural mock.

Deliberately uses the *adjacent-pair* RoPE convention
(`[x0, x1] -> [-x1, x0]`), not HuggingFace's rotate-half (contiguous-half)
convention -- this is the concrete case `PretrainArchitectureAdapter`'s
docstring is about: an existing HF-oriented attention bridge would silently
apply the wrong rotation to a model using this convention.

Also deliberately computes RoPE `cos`/`sin` once per forward pass and
passes them into every block call as extra positional args
(`block(x, cos, sin)`), matching the target architecture's convention.
This matters beyond efficiency: `BlockBridge` wraps a bare-tensor output
in a 1-tuple for single-positional-argument "standalone hidden_states"
calls (an HF-compatibility convention). A loop calling `block(x)` alone
would need to unwrap `[0]` from each result -- modifying the source
model's forward specifically because it's being bridged. Passing
`cos`/`sin` positionally avoids that, so the forward loop needs no
changes at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_adjacent_pairs(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_adjacent_pairs(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return x * cos + _rotate_adjacent_pairs(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class Attention(nn.Module):
    """Receives precomputed `cos`/`sin` rather than computing them
    internally -- matches the real target architecture, where rotary
    tables are computed once per forward pass and shared across blocks."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [b, n_heads, s, d_head]

        q = _apply_rotary_adjacent_pairs(q, cos, sin)
        k = _apply_rotary_adjacent_pairs(k, cos, sin)

        attn = torch.softmax(
            (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
            + torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1),
            dim=-1,
        )
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.o(out)


class DenseMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([DenseMLP(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.router(x), dim=-1)
        topk_weights, topk_idx = weights.topk(self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            mask = topk_idx == e
            if not mask.any():
                continue
            gate = (mask.float() * topk_weights).sum(dim=-1, keepdim=True)
            out = out + gate * self.experts[e](x)
        return out


class Block(nn.Module):
    """Takes `cos`/`sin` as extra positional args alongside the hidden
    state -- see the module docstring for why this matters beyond RoPE
    plumbing."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, mlp: nn.Module):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = mlp

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyPretrainModel(nn.Module):
    """Full tiny reference model for numerical-parity integration tests.
    Matches `PretrainArchitectureAdapter`'s expected attribute paths:
    `embed`, `blocks`, `norm_f`, `lm_head`.
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
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    d_ff,
                    mlp=(
                        MoEMLP(d_model, d_ff, n_experts, top_k)
                        if i in moe_layer_indices
                        else DenseMLP(d_model, d_ff)
                    ),
                )
                for i in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

        d_head = d_model // n_heads
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, tokens: torch.Tensor) -> dict:
        b, s = tokens.shape
        x = self.embed(tokens)

        # Computed once per forward pass, passed into every block -- the
        # real target architecture's convention (see module docstring).
        pos = torch.arange(s, device=tokens.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        cos, sin = freqs.cos(), freqs.sin()

        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        return {"logits": self.lm_head(x)}
