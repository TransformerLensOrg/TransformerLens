"""The wrapped HF model must never enter the bridge's registered module tree.

``nn.Module.__setattr__`` claims Module values before any data descriptor runs, so an
ordinary ``self.original_model = ...`` silently registers it as a submodule: composed
``state_dict`` gains aliased ``original_model.*`` keys and ``.to()`` moves the weights twice.
"""
from __future__ import annotations

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.model_bridge.sources.native import (
    NativeModel,
    initialize_native_model,
)


def _build_cfg(**overrides) -> TransformerBridgeConfig:
    base = dict(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        architecture="TransformerLensNative",
        seed=0,
    )
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def _bridge() -> TransformerBridge:
    cfg = _build_cfg()
    model = NativeModel(cfg)
    initialize_native_model(model, cfg)
    return build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)


def _assert_unregistered(bridge: TransformerBridge, when: str) -> None:
    assert "original_model" in bridge.__dict__, f"{when}: dropped out of __dict__"
    assert "original_model" not in bridge._modules, f"{when}: registered as a submodule"
    assert [
        name for name, _ in bridge.named_modules() if name.split(".")[0] == "original_model"
    ] == [], f"{when}: appears in named_modules()"
    assert [
        key for key in bridge.state_dict() if key.startswith("original_model.")
    ] == [], f"{when}: aliased into state_dict"


def test_wrapped_model_is_unregistered_after_boot():
    _assert_unregistered(_bridge(), "after boot")


def test_dtype_move_does_not_register_wrapped_model():
    bridge = _bridge()
    before = len(bridge.state_dict())

    bridge.to(torch.float64)

    _assert_unregistered(bridge, "after .to(float64)")
    assert len(bridge.state_dict()) == before, "state_dict grew during the move"


def test_dtype_move_still_reaches_the_wrapped_model():
    """Guards against 'fixing' registration by skipping the move entirely."""
    bridge = _bridge()
    assert next(bridge.original_model.parameters()).dtype == torch.float32

    bridge.to(torch.float64)

    assert next(bridge.original_model.parameters()).dtype == torch.float64


def test_hf_attribute_delegation_survives_a_move():
    """Delegation reads __dict__, so registration silently breaks it."""
    bridge = _bridge()
    bridge.to(torch.float64)

    assert bridge.original_model is bridge.__dict__["original_model"]
