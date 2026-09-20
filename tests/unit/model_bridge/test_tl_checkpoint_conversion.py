"""Tests for the legacy TL-property-format checkpoint converter (#1588).

Historical HookedTransformer checkpoints (OthelloGPT, grokking, ARENA content)
are saved with the old property-style keys ("blocks.0.attn.W_Q", "embed.W_E",
...) and per-head tensor shapes. `convert_tl_checkpoint` maps those onto the
key/tensor format `TransformerBridge.boot_native(cfg).load_state_dict` accepts
natively, so these checkpoints can be loaded once and re-saved in bridge
format without teaching `load_state_dict` a second key convention.

Inputs are FROZEN fixtures under tests/fixtures/tl_checkpoints/ (captured by
scripts/capture_tl_checkpoint_fixtures.py): the legacy format never changes —
that is the converter's whole premise — so its test inputs are captured once
from a real HookedTransformer, with that model's logits as the behavioral
reference, and survive the class's 4.0 deletion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.tl_checkpoint_conversion import convert_tl_checkpoint

FIXTURES = Path(__file__).parents[2] / "fixtures" / "tl_checkpoints"


def _load(variant: str):
    """(legacy state_dict, reference dict, TransformerBridgeConfig) for a variant."""
    root = FIXTURES / variant
    checkpoint = torch.load(root / "checkpoint.pt", weights_only=True)
    reference = torch.load(root / "reference.pt", weights_only=True)
    kwargs = json.loads((root / "meta.json").read_text())
    return checkpoint, reference, TransformerBridgeConfig(**kwargs)


def _converted_bridge(variant: str, strict: bool = True):
    checkpoint, reference, bridge_cfg = _load(variant)
    converted = convert_tl_checkpoint(checkpoint, bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=strict)
    return bridge, reference, result


def _assert_forward_matches(bridge, reference):
    with torch.no_grad():
        bridge_logits = bridge(reference["tokens"])
    torch.testing.assert_close(bridge_logits, reference["logits"], atol=1e-4, rtol=1e-4)


def test_convert_tl_checkpoint_loads_strict_into_native_bridge():
    _, _, result = _converted_bridge("default")
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_convert_tl_checkpoint_matches_source_forward_pass():
    bridge, reference, _ = _converted_bridge("default")
    _assert_forward_matches(bridge, reference)


def test_convert_tl_checkpoint_places_qkvo_in_correct_head_slots():
    bridge, reference, _ = _converted_bridge("default")
    for attr in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"):
        torch.testing.assert_close(getattr(bridge, attr), reference[attr])


def test_convert_tl_checkpoint_raises_on_cfg_mismatch():
    checkpoint, _, _ = _load("default")
    wrong_cfg = TransformerBridgeConfig(
        **{
            **json.loads((FIXTURES / "default" / "meta.json").read_text()),
            "n_heads": 4,
            "d_head": 8,
        }
    )
    with pytest.raises(ValueError, match="head|shape|mismatch"):
        convert_tl_checkpoint(checkpoint, wrong_cfg)


def test_convert_tl_checkpoint_raises_on_unrecognized_key():
    checkpoint, _, bridge_cfg = _load("default")
    checkpoint["blocks.0.attn.W_mystery"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="W_mystery|nrecognized"):
        convert_tl_checkpoint(checkpoint, bridge_cfg)


def test_convert_tl_checkpoint_supports_gqa():
    bridge, reference, result = _converted_bridge("gqa")
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    _assert_forward_matches(bridge, reference)


def test_convert_tl_checkpoint_supports_lnpre():
    bridge, reference, result = _converted_bridge("lnpre")
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    _assert_forward_matches(bridge, reference)


def test_convert_tl_checkpoint_supports_attn_only():
    bridge, reference, result = _converted_bridge("attn_only")
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    _assert_forward_matches(bridge, reference)


def test_convert_tl_checkpoint_drops_attention_buffers():
    checkpoint, _, bridge_cfg = _load("default")
    assert any(
        key.endswith((".mask", ".IGNORE")) for key in checkpoint
    ), "fixture lost the legacy attention buffers; recapture it"
    converted = convert_tl_checkpoint(checkpoint, bridge_cfg)
    assert not any(key.endswith((".mask", ".IGNORE")) for key in converted)


def test_convert_tl_checkpoint_supports_gated_mlp_and_rms_norm():
    """The native bridge's gated MLP has no bias parameter at all for
    gate/in/out (matching real gated-MLP HF architectures like Llama) while
    HookedTransformer's gated MLP kept live b_in/b_out parameters. The
    converter still faithfully translates those keys since they are real
    checkpoint parameters; load_state_dict refuses them under strict=True.
    In the fixture they are exactly zero (freshly constructed, untrained), so
    dropping them via strict=False is lossless and the forward still matches.
    """
    checkpoint, reference, bridge_cfg = _load("gated_rms")
    converted = convert_tl_checkpoint(checkpoint, bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=False)

    assert result.missing_keys == []
    assert set(result.unexpected_keys) == {
        f"blocks.{i}.mlp.{part}.bias" for i in range(bridge_cfg.n_layers) for part in ("in", "out")
    }
    _assert_forward_matches(bridge, reference)
