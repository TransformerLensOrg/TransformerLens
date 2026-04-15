"""Regression test: attention hooks fire on forward (C1 guard).

Complements the alias-resolution test: aliases can resolve yet the HookPoint
may never fire if forward bypasses the LinearBridge (the original C1 bug).
Isolates blocks.0.attn per PositionEmbeddingsAttentionBridge adapter with a
synthetic HF module and asserts hook_q/k/v/z all fire.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


def _stub_cfg(architecture: str, **kw: Any) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_vocab=1000,
        d_mlp=256,
        n_key_value_heads=kw.get("n_key_value_heads", 4),
        default_prepend_bos=True,
        architecture=architecture,
    )


def _position_embeddings_adapters() -> list[str]:
    """Adapters using PositionEmbeddingsAttentionBridge directly (not subclasses)."""
    results: list[str] = []
    for arch, adapter_cls in sorted(SUPPORTED_ARCHITECTURES.items()):
        try:
            adapter = adapter_cls(_stub_cfg(arch))
        except Exception:
            continue
        mapping = adapter.component_mapping
        if mapping is None or "blocks" not in mapping:
            continue
        attn = (mapping["blocks"].submodules or {}).get("attn")
        # type() check excludes JointQKV subclasses.
        if attn is not None and type(attn) is PositionEmbeddingsAttentionBridge:
            results.append(arch)
    return results


class _FakeHFAttn(nn.Module):
    """Synthetic HF attention module with q/k/v/o + optional QK-norms."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        with_q_norm: bool,
        with_k_norm: bool,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = n_heads // n_kv_heads if n_kv_heads else 1
        self.scaling = head_dim**-0.5
        self.attention_dropout = 0.0
        self.layer_idx = 0
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        if with_q_norm:
            self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=True)
        if with_k_norm:
            self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=True)


def _make_fake_hf_attn(
    attn_bridge: PositionEmbeddingsAttentionBridge,
    d_model: int,
    n_heads: int,
    head_dim: int,
    n_kv_heads: int,
) -> nn.Module:
    return _FakeHFAttn(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads,
        with_q_norm="q_norm" in attn_bridge.submodules,
        with_k_norm="k_norm" in attn_bridge.submodules,
    )


def _wire_attention_submodules(
    attn_bridge: PositionEmbeddingsAttentionBridge, fake_hf_attn: nn.Module
) -> None:
    """Mirror setup_components: set original_component + add_module for each sub."""
    for name, sub in attn_bridge.submodules.items():
        hf_sub = getattr(fake_hf_attn, name + "_proj", None)
        if hf_sub is None:
            hf_sub = getattr(fake_hf_attn, name, None)
        if hf_sub is None:
            raise RuntimeError(f"fake HF attn missing '{name}' (tried {name}_proj, {name})")
        sub.set_original_component(hf_sub)
        if name not in attn_bridge._modules:
            attn_bridge.add_module(name, sub)


_CRITICAL_HOOKS = {"hook_q", "hook_k", "hook_v", "hook_z"}


@pytest.mark.parametrize("architecture", _position_embeddings_adapters())
def test_attention_critical_hooks_fire_on_forward(architecture: str) -> None:
    """Assert hook_q/k/v/z fire during attention forward (C1 regression guard)."""
    adapter_cls = SUPPORTED_ARCHITECTURES[architecture]
    adapter = adapter_cls(_stub_cfg(architecture))
    attn = adapter.component_mapping["blocks"].submodules["attn"]
    assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    d_model = adapter.cfg.d_model
    n_heads = adapter.cfg.n_heads
    head_dim = d_model // n_heads
    n_kv_heads = getattr(adapter.cfg, "n_key_value_heads", n_heads) or n_heads

    fake_hf_attn = _make_fake_hf_attn(attn, d_model, n_heads, head_dim, n_kv_heads)
    try:
        attn.set_original_component(fake_hf_attn)
    except RuntimeError as e:
        pytest.skip(f"{architecture}: cannot wire synthetic HF attn ({e})")

    _wire_attention_submodules(attn, fake_hf_attn)

    fired: set[str] = set()
    handles = []
    for hname, hp in attn.get_hooks().items():
        h = hp.add_hook(lambda t, hook, n=hname: (fired.add(n), t)[1])
        handles.append(h)

    # Record alias-targeted hooks under their alias name (hook_q -> q.hook_out).
    for alias_name, target in attn.hook_aliases.items():
        if isinstance(target, str):
            try:
                obj: Any = attn
                for part in target.split("."):
                    obj = (
                        obj.submodules.get(part)
                        if hasattr(obj, "submodules") and part in obj.submodules
                        else getattr(obj, part)
                    )
                if hasattr(obj, "add_hook"):
                    obj.add_hook(lambda t, hook, n=alias_name: (fired.add(n), t)[1])
            except AttributeError:
                pass

    batch, seq = 1, 4
    hidden = torch.randn(batch, seq, d_model)
    cos = torch.ones(1, seq, head_dim)
    sin = torch.zeros(1, seq, head_dim)
    attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)

    missing = _CRITICAL_HOOKS - fired
    assert not missing, (
        f"{architecture}: critical attention hooks did not fire: {sorted(missing)}. "
        f"Fired: {sorted(fired)}"
    )
