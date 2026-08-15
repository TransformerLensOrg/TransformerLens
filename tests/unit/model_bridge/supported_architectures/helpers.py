"""Shared helpers for per-adapter unit tests."""

from __future__ import annotations

import copy
from typing import Any

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.component_setup import setup_submodules
from transformer_lens.model_bridge.generalized_components import MoEBridge

# The keys MoEBridge binds a dense layer's projections under. Adapter key-set
# assertions subtract these and check only the sparse-side keys: that every
# dense-aware adapter declares dense_* AND that each key reads the projection it
# names is covered behaviorally, for all 14 of them, by the roster in
# tests/unit/model_bridge/generalized_components/test_moe_dense_dispatch.py.
DENSE_KEYS = frozenset({*MoEBridge.DENSE_SUBMODULE_KEYS, MoEBridge.DENSE_GATE_KEY})


def make_bridge_cfg(architecture: str, **overrides) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig with the standard tiny test dims.

    Defaults: d_model=64, n_heads=8, n_layers=2, d_vocab=100, n_ctx=128,
    default_prepend_bos=False. d_head is derived from d_model/n_heads unless
    overridden. Pass any TransformerBridgeConfig field as a keyword.
    """
    cfg = dict(
        d_model=64,
        n_heads=8,
        n_layers=2,
        n_ctx=128,
        d_vocab=100,
        default_prepend_bos=False,
        architecture=architecture,
    )
    cfg.update(overrides)
    cfg.setdefault("d_head", cfg["d_model"] // cfg["n_heads"])
    return TransformerBridgeConfig(**cfg)


def make_additive_causal_mask(batch: int, seq_len: int) -> torch.Tensor:
    """4D additive causal mask — the form HF eager and the bridges add as-is."""
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = torch.zeros(batch, 1, seq_len, seq_len)
    return mask.masked_fill(~causal, torch.finfo(torch.float32).min)


def identity_rope(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity rotary embeddings (cos=1, sin=0) shaped [1, seq_len, head_dim].

    Identity only under that broadcast shape — callers must not reshape.
    """
    return torch.ones(1, seq_len, head_dim), torch.zeros(1, seq_len, head_dim)


def wire_attention_bridge(
    adapter: Any,
    hf_attn: Any,
    component: str = "blocks.0.attn",
    expected_type: type | None = None,
) -> Any:
    """Deepcopy the adapter's attention-bridge template and wire hf_attn into it
    the way component setup does (original component, submodules, hook compat)."""
    bridge = copy.deepcopy(adapter.get_generalized_component(component))
    if expected_type is not None:
        assert isinstance(bridge, expected_type)
    bridge.set_original_component(hf_attn)
    setup_submodules(bridge, adapter, hf_attn)
    bridge.setup_hook_compatibility()
    return bridge


class FakeDelegatedAttention(torch.nn.Module):
    """HF-shaped attention for delegated bridges: routes through every projection
    (hooks fire inside the wrapped submodules) and returns (out, weights[B,H,S,S]).
    o_proj takes n_heads*d_head — a narrower fake silently no-ops the hook_z reshape.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.d_head = n_heads, n_kv_heads, d_head
        self.q_proj = torch.nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = torch.nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = torch.nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * d_head, d_model, bias=False)
        for name in ("q_norm", "k_norm", "v_norm"):
            setattr(self, name, torch.nn.Identity())

    def forward(self, hidden_states, *args, **kwargs):
        batch, seq, _ = hidden_states.shape
        self.q_norm(self.q_proj(hidden_states))
        self.k_norm(self.k_proj(hidden_states))
        value = self.v_norm(self.v_proj(hidden_states))
        expanded = value.view(batch, seq, self.n_kv_heads, self.d_head).repeat_interleave(
            self.n_heads // self.n_kv_heads, dim=2
        )
        weights = torch.softmax(torch.zeros(batch, self.n_heads, seq, seq), dim=-1)
        return self.o_proj(expanded.reshape(batch, seq, -1)), weights
