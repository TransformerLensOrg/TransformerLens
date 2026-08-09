"""Unit tests for stop_at_layer and input_to_embed on native Bridge.

Native Bridge blocks are named 'layers.N' (top-level, no preceding dot).
_extract_layer_idx must parse that pattern so stop_at_layer actually stops.
Regression for: https://github.com/TransformerLensOrg/TransformerLens/issues/1632
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _bridge(n_layers: int = 3) -> TransformerBridge:
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=n_layers,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


TOKENS = torch.tensor([[1, 2, 3]])


def test_native_block_names_are_layers_pattern():
    bridge = _bridge()
    names = [block.name for block in bridge.blocks]
    assert names == ["layers.0", "layers.1", "layers.2"], names


@pytest.mark.parametrize("stop_at_layer", [0, 1, 2])
def test_stop_at_layer_returns_residual_stream(stop_at_layer):
    bridge = _bridge()
    out = bridge(TOKENS, stop_at_layer=stop_at_layer)
    assert out.shape == (1, 3, 32), (
        f"stop_at_layer={stop_at_layer}: expected residual stream shape (1,3,32), got {out.shape}"
    )


def test_stop_at_layer_negative_returns_residual_stream():
    bridge = _bridge()
    out = bridge(TOKENS, stop_at_layer=-1)
    assert out.shape == (1, 3, 32), f"stop_at_layer=-1: expected (1,3,32), got {out.shape}"


def test_stop_at_layer_0_differs_from_stop_at_layer_1():
    bridge = _bridge()
    out0 = bridge(TOKENS, stop_at_layer=0)
    out1 = bridge(TOKENS, stop_at_layer=1)
    assert not torch.allclose(out0, out1), "stop_at_layer=0 and 1 should differ (block 0 runs for 1)"


def test_input_to_embed_returns_residual_stream_not_logits():
    bridge = _bridge()
    residual, _, _, _ = bridge.input_to_embed(TOKENS)
    assert residual.shape == (1, 3, 32), (
        f"input_to_embed should return d_model=32, got {residual.shape} — "
        "if shape is (1,3,16) the logits are leaking through stop_at_layer=0"
    )
