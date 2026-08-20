"""Tests for direct assignment of Bridge-managed hook flags (#1689)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def _cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )


def _tiny_gpt2_bridge() -> TransformerBridge:
    hf_config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=32,
        n_positions=8,
        n_ctx=8,
        vocab_size=16,
    )
    hf_model = GPT2LMHeadModel(hf_config).eval()
    return build_bridge_from_module(
        hf_model,
        "GPT2LMHeadModel",
        hf_config=hf_config,
        tokenizer=None,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("flag_name", "hook_name"),
    [
        ("use_attn_result", "blocks.0.attn.hook_result"),
        ("use_attn_in", "blocks.0.attn.hook_attn_in"),
        ("use_hook_mlp_in", "blocks.0.hook_mlp_in"),
        ("use_split_qkv_input", "blocks.0.attn.hook_q_input"),
    ],
)
def test_direct_assignment_matches_setter_hook_behavior(flag_name: str, hook_name: str) -> None:
    bridge = _tiny_gpt2_bridge()
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 8))

    setattr(bridge.cfg, flag_name, True)
    _, direct_cache = bridge.run_with_cache(tokens, names_filter=[hook_name])

    setattr(bridge.cfg, flag_name, False)
    getattr(bridge, f"set_{flag_name}")(True)
    _, setter_cache = bridge.run_with_cache(tokens, names_filter=[hook_name])

    assert list(direct_cache) == [hook_name]
    assert list(setter_cache) == [hook_name]
    assert direct_cache[hook_name].shape == setter_cache[hook_name].shape


def test_direct_assignment_preserves_mutual_exclusivity() -> None:
    bridge = _tiny_gpt2_bridge()

    bridge.cfg.use_split_qkv_input = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        bridge.cfg.use_attn_in = True
    assert bridge.cfg.use_attn_in is False

    bridge.cfg.use_split_qkv_input = False
    bridge.cfg.use_attn_in = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        bridge.cfg.use_split_qkv_input = True
    assert bridge.cfg.use_split_qkv_input is False


@pytest.mark.parametrize("flag_name", ["use_attn_result", "use_attn_in", "use_split_qkv_input"])
def test_direct_assignment_preserves_unsupported_architecture_errors(
    monkeypatch: pytest.MonkeyPatch, flag_name: str
) -> None:
    bridge = TransformerBridge.boot_native(_cfg())

    class _FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.Identity()

    monkeypatch.setattr(bridge, "blocks", nn.ModuleList([_FakeBlock()]), raising=True)

    with pytest.raises(NotImplementedError, match=flag_name):
        setattr(bridge.cfg, flag_name, True)
    assert getattr(bridge.cfg, flag_name) is False


def test_deepcopied_live_config_is_not_bound_to_original_bridge() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    copied_cfg = copy.deepcopy(bridge.cfg)

    copied_cfg.use_hook_mlp_in = True

    assert copied_cfg.use_hook_mlp_in is True
    assert bridge.cfg.use_hook_mlp_in is False


def test_deepcopied_bridge_rebinds_its_config() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    copied_bridge = copy.deepcopy(bridge)

    copied_bridge.cfg.use_hook_mlp_in = True

    assert copied_bridge.cfg.use_hook_mlp_in is True
    assert copied_bridge.blocks[0].config.use_hook_mlp_in is True
    assert bridge.cfg.use_hook_mlp_in is False
