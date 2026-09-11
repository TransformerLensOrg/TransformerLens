"""A second process_weights() call warns and skips instead of re-folding.

Folding reads its factors out of the weights and neutralizes them, so a repeat pass is
a no-op for standard adapters. Adapters that hold a fold factor outside the weights
would double-apply it, and re-executing a notebook cell makes that routine.
"""
from __future__ import annotations

import logging

import torch

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


def test_second_call_warns_and_skips(caplog) -> None:
    bridge = _bridge()
    bridge.process_weights()

    with caplog.at_level(logging.WARNING):
        bridge.process_weights()

    assert "already applied" in caplog.text


def test_second_call_leaves_weights_untouched() -> None:
    bridge = _bridge()
    bridge.process_weights()
    after_first = {key: value.clone() for key, value in bridge.state_dict().items()}

    bridge.process_weights()

    after_second = bridge.state_dict()
    assert set(after_second) == set(after_first)
    for key, value in after_first.items():
        torch.testing.assert_close(after_second[key], value, atol=0.0, rtol=0.0, msg=key)


def test_first_call_still_processes() -> None:
    """Guards against skipping the very first call."""
    bridge = _bridge()
    before = {key: value.clone() for key, value in bridge.state_dict().items()}

    bridge.process_weights()

    after = bridge.state_dict()
    assert any(
        not torch.equal(after[key], value) for key, value in before.items()
    ), "process_weights changed nothing on a fresh bridge"


def test_logits_are_unchanged_by_the_second_call() -> None:
    bridge = _bridge()
    tokens = torch.randint(0, 16, (2, 8))
    bridge.process_weights()
    first = bridge(tokens, return_type="logits")

    bridge.process_weights()

    torch.testing.assert_close(bridge(tokens, return_type="logits"), first)
