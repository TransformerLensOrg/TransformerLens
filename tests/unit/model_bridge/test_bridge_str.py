"""``print(model)`` is the first debugging move, so ``__str__`` must not raise.

The helper it used to call was deleted while the call site stayed, so every bridge raised
AttributeError here for months — the surface had no test at all.
"""
from __future__ import annotations

import copy

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.model_bridge.sources.native import (
    NativeModel,
    initialize_native_model,
)


def _bridge() -> TransformerBridge:
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
        architecture="TransformerLensNative",
        seed=0,
    )
    model = NativeModel(cfg)
    initialize_native_model(model, cfg)
    return build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)


def test_str_does_not_raise():
    text = str(_bridge())

    assert text.startswith("TransformerBridge:")
    assert len(text.splitlines()) > 1, "summary listed no components"


def test_str_names_top_level_components():
    bridge = _bridge()
    text = str(bridge)

    for name in bridge.adapter.get_component_mapping():
        assert f"{name}: " in text, f"component {name!r} missing from the summary"


def test_str_indents_submodules_below_their_parent():
    lines = str(_bridge()).splitlines()

    indented = [line for line in lines[1:] if line.startswith("    ")]
    assert indented, "no submodule was rendered below its parent"


def test_str_works_on_a_shallow_copy():
    assert str(copy.copy(_bridge())).startswith("TransformerBridge:")
