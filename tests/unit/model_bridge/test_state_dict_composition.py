"""Regression tests for recursive TransformerBridge checkpoint composition (#1655)."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    JointGateUpMLPBridge,
    JointQKVAttentionBridge,
    LinearBridge,
)
from transformer_lens.model_bridge.sources import build_bridge_from_module


def _native_bridge() -> TransformerBridge:
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=2,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


def _parent_with_bridge(bridge: TransformerBridge) -> torch.nn.Module:
    parent = torch.nn.Module()
    parent.add_module("bridge", bridge)
    return parent


def test_state_dict_with_destination_and_prefix_uses_recursive_semantics() -> None:
    bridge = _native_bridge()
    sentinel = torch.tensor(1)
    destination: OrderedDict[str, torch.Tensor] = OrderedDict({"sentinel": sentinel})

    returned = bridge.state_dict(destination=destination, prefix="nested.bridge.")

    assert returned is destination
    assert destination["sentinel"] is sentinel
    recursive_keys = set(destination) - {"sentinel"}
    assert recursive_keys
    assert all(key.startswith("nested.bridge.") for key in recursive_keys)


def test_parent_state_dict_strict_round_trip() -> None:
    parent = _parent_with_bridge(_native_bridge())
    checkpoint = {key: value.clone() for key, value in parent.state_dict().items()}

    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.zero_()

    result = parent.load_state_dict(checkpoint, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    reloaded = parent.state_dict()
    for key, value in checkpoint.items():
        assert torch.equal(reloaded[key], value), f"{key} did not round-trip"


def test_parent_registration_is_stable_across_first_forward() -> None:
    bridge = _native_bridge()
    parent = _parent_with_bridge(bridge)

    for block in bridge.blocks:
        assert block.attn._ln1_module is block.ln1.original_component

    keys_before = tuple(parent.state_dict())
    assert not any("._ln1_module." in key for key in keys_before)
    with torch.no_grad():
        bridge(torch.randint(0, bridge.cfg.d_vocab, (1, 4)))
    keys_after = tuple(parent.state_dict())

    assert keys_after == keys_before
    assert not any("._ln1_module." in key for key in keys_after)


def test_nested_joint_qkv_bridge_strict_round_trip() -> None:
    cfg = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_inner=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    bridge = build_bridge_from_module(
        GPT2LMHeadModel(cfg),
        architecture="GPT2LMHeadModel",
        hf_config=cfg,
    )
    parent = _parent_with_bridge(bridge)

    checkpoint = {key: value.clone() for key, value in parent.state_dict().items()}
    assert not any(".qkv." in key for key in checkpoint)
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.zero_()

    result = parent.load_state_dict(checkpoint, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    reloaded = parent.state_dict()
    for key, value in checkpoint.items():
        assert torch.equal(reloaded[key], value), f"{key} did not round-trip"


def _filtered_joint_component(kind: str) -> torch.nn.Module:
    filtered_child = LinearBridge(name=kind)
    filtered_child.set_original_component(torch.nn.Linear(4, 8))
    cfg = SimpleNamespace(n_heads=2, d_head=4)

    if kind == "qkv":
        qkv_component = JointQKVAttentionBridge(
            name="attn",
            config=cfg,
            submodules={"qkv": filtered_child},
        )
        for child_name in ("q", "k", "v"):
            getattr(qkv_component, child_name).set_original_component(torch.nn.Linear(4, 4))
        return qkv_component
    gate_up_component = JointGateUpMLPBridge(
        name="mlp",
        config=cfg,
        submodules={"gate_up": filtered_child},
    )
    gate_up_component.add_module("gate_up", filtered_child)
    gate_up_component.gate.set_original_component(torch.nn.Linear(4, 4))
    getattr(gate_up_component, "in").set_original_component(torch.nn.Linear(4, 4))
    return gate_up_component


@pytest.mark.parametrize("filtered_child_name", ["qkv", "gate_up"])
def test_filtered_joint_component_strict_round_trip(filtered_child_name: str) -> None:
    component = _filtered_joint_component(filtered_child_name)
    filtered_child = component.get_submodule(filtered_child_name)
    checkpoint = {key: value.clone() for key, value in component.state_dict().items()}

    assert checkpoint
    assert not any(key.startswith(f"{filtered_child_name}.") for key in checkpoint)
    with torch.no_grad():
        for parameter in component.parameters():
            parameter.zero_()

    result = component.load_state_dict(checkpoint, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    reloaded = component.state_dict()
    for key, value in checkpoint.items():
        assert torch.equal(reloaded[key], value), f"{key} did not round-trip"
    for parameter in filtered_child.parameters():
        assert torch.count_nonzero(parameter) == 0
