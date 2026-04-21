"""Component-level tests for per-head attention hooks.

No HF model load: synthesizes a minimal HF attention module and wires it into
`PositionEmbeddingsAttentionBridge` / `JointQKVAttentionBridge` directly so the
tests run fast and cover GQA (n_kv_heads < n_heads) on a PEA adapter where no
HF fixture is cheap enough for CI.

Covers:
- `hook_result` (per-head output pre-sum) shape and
  `sum(hook_result, dim=heads) + b_O == hook_out`
- `hook_q_input`/`hook_k_input`/`hook_v_input` shape (H, n_kv_heads for K/V
  under GQA)
- Independence guarantee: patching `hook_q_input` leaves K and V unchanged
- `hook_attn_in` shape and firing under `use_attn_in`
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


def _cfg(n_heads: int = 4, n_kv_heads: int | None = None) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=1,
        n_ctx=32,
        n_heads=n_heads,
        d_vocab=100,
        d_mlp=128,
        n_key_value_heads=n_kv_heads if n_kv_heads is not None else n_heads,
        default_prepend_bos=True,
        architecture="llama",  # non-Qwen3.5 to avoid gated_q_proj
    )


class _FakeRoPEHFAttn(nn.Module):
    """Minimal HF-like attention module — q/k/v/o_proj + head_dim."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, n_kv_heads: int):
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


def _wire_pea(
    cfg: TransformerBridgeConfig,
) -> tuple[PositionEmbeddingsAttentionBridge, _FakeRoPEHFAttn]:
    from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

    attn = PositionEmbeddingsAttentionBridge(
        name="self_attn",
        config=cfg,
        submodules={
            "q": LinearBridge(name="q_proj"),
            "k": LinearBridge(name="k_proj"),
            "v": LinearBridge(name="v_proj"),
            "o": LinearBridge(name="o_proj"),
        },
    )
    hf = _FakeRoPEHFAttn(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        head_dim=cfg.d_head,
        n_kv_heads=cfg.n_key_value_heads or cfg.n_heads,
    )
    attn.set_original_component(hf)
    for name in ("q", "k", "v", "o"):
        sub = attn.submodules[name]
        sub.set_original_component(getattr(hf, f"{name}_proj"))
        if name not in attn._modules:
            attn.add_module(name, sub)
    return attn, hf


def _rope_inputs(cfg: TransformerBridgeConfig, batch: int = 1, seq: int = 4):
    hidden = torch.randn(batch, seq, cfg.d_model)
    cos = torch.ones(1, seq, cfg.d_head)
    sin = torch.zeros(1, seq, cfg.d_head)
    return hidden, cos, sin


@pytest.mark.parametrize("n_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_hook_result_shape_and_sum(n_kv_heads: int) -> None:
    cfg = _cfg(n_heads=4, n_kv_heads=n_kv_heads)
    attn, _ = _wire_pea(cfg)
    cfg.use_attn_result = True
    hidden, cos, sin = _rope_inputs(cfg)

    captured: dict = {}

    def cap(name):
        def _hook(tensor, hook):
            captured[name] = tensor.detach().clone()
            return tensor

        return _hook

    attn.hook_result.add_hook(cap("result"))
    attn.hook_out.add_hook(cap("out"))
    attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)

    assert "result" in captured, "hook_result must fire when use_attn_result=True"
    assert captured["result"].shape == (1, 4, cfg.n_heads, cfg.d_model), (
        f"hook_result shape {tuple(captured['result'].shape)} != "
        f"(1, 4, {cfg.n_heads}, {cfg.d_model})"
    )
    # b_O is None for the fake module (bias=False).
    summed = captured["result"].sum(dim=-2)
    assert torch.allclose(summed, captured["out"], atol=1e-5)


def test_hook_result_does_not_fire_when_flag_off() -> None:
    cfg = _cfg(n_heads=4)
    attn, _ = _wire_pea(cfg)
    cfg.use_attn_result = False
    hidden, cos, sin = _rope_inputs(cfg)

    fired = {"result": False}

    def _hook(tensor, hook):
        fired["result"] = True
        return tensor

    attn.hook_result.add_hook(_hook)
    attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)
    assert fired["result"] is False


@pytest.mark.parametrize("n_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_split_qkv_input_shapes_and_independence(n_kv_heads: int) -> None:
    cfg = _cfg(n_heads=4, n_kv_heads=n_kv_heads)
    attn, _ = _wire_pea(cfg)
    cfg.use_split_qkv_input = True
    hidden, cos, sin = _rope_inputs(cfg)

    captured: dict = {}

    def cap(name):
        def _hook(tensor, hook):
            captured[name] = tensor.detach().clone()
            return tensor

        return _hook

    attn.hook_q_input.add_hook(cap("q_in"))
    attn.hook_k_input.add_hook(cap("k_in"))
    attn.hook_v_input.add_hook(cap("v_in"))
    # Per-head projection output.
    attn.submodules["k"].hook_out.add_hook(cap("k_out_baseline"))
    attn.submodules["v"].hook_out.add_hook(cap("v_out_baseline"))
    attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)

    assert captured["q_in"].shape == (1, 4, cfg.n_heads, cfg.d_model)
    assert captured["k_in"].shape == (1, 4, cfg.n_key_value_heads, cfg.d_model)
    assert captured["v_in"].shape == (1, 4, cfg.n_key_value_heads, cfg.d_model)

    # Independence: patch hook_q_input to zeros, rerun, K/V must not change.
    for hp in (attn.hook_q_input, attn.hook_k_input, attn.hook_v_input):
        hp.remove_hooks()
    attn.submodules["k"].hook_out.remove_hooks()
    attn.submodules["v"].hook_out.remove_hooks()

    patched: dict = {}

    def zero_q(tensor, hook):
        return torch.zeros_like(tensor)

    attn.hook_q_input.add_hook(zero_q)
    attn.submodules["k"].hook_out.add_hook(cap("k_out_patched"))
    attn.submodules["v"].hook_out.add_hook(cap("v_out_patched"))
    # Inject into captured-dict at new keys so baselines stay.
    captured_patched: dict = {}

    def cap_p(name):
        def _hook(tensor, hook):
            captured_patched[name] = tensor.detach().clone()
            return tensor

        return _hook

    attn.submodules["k"].hook_out.remove_hooks()
    attn.submodules["v"].hook_out.remove_hooks()
    attn.submodules["k"].hook_out.add_hook(cap_p("k_out"))
    attn.submodules["v"].hook_out.add_hook(cap_p("v_out"))
    attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)

    assert torch.equal(
        captured["k_out_baseline"], captured_patched["k_out"]
    ), "Patching hook_q_input altered k projection output"
    assert torch.equal(
        captured["v_out_baseline"], captured_patched["v_out"]
    ), "Patching hook_q_input altered v projection output"


def test_attn_in_shape() -> None:
    cfg = _cfg(n_heads=4)
    attn, _ = _wire_pea(cfg)
    cfg.use_attn_in = True
    hidden, cos, sin = _rope_inputs(cfg)

    captured: dict = {}

    def cap(name):
        def _hook(tensor, hook):
            captured[name] = tensor.detach().clone()
            return tensor

        return _hook

    attn.hook_attn_in.add_hook(cap("attn_in"))
    attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)
    assert captured["attn_in"].shape == (1, 4, cfg.n_heads, cfg.d_model)


def test_gated_q_proj_raises_when_split_active() -> None:
    cfg = _cfg(n_heads=4)
    cfg.gated_q_proj = True
    cfg.use_split_qkv_input = True
    attn, _ = _wire_pea(cfg)
    hidden, cos, sin = _rope_inputs(cfg)
    with pytest.raises(NotImplementedError, match="gated q_proj"):
        attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)
