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
