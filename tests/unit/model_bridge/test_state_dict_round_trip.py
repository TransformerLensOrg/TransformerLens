"""Regression tests for TransformerBridge.state_dict()/load_state_dict() round-tripping (#1587).

state_dict() emits TL-renamed keys (e.g. "blocks.0.attn.q.weight"), but
load_state_dict() only matched raw native parameter names, so a
state_dict() -> load_state_dict() round trip silently loaded nothing and
strict=True was silently downgraded to strict=False.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _native_cfg(**overrides) -> TransformerBridgeConfig:
    base = dict(
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
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def test_native_round_trip_overwrites_params_not_a_noop():
    bridge = TransformerBridge.boot_native(_native_cfg())

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    assert sd, "state_dict() returned no TL-format keys"

    with torch.no_grad():
        for p in bridge.parameters():
            p.zero_()
    assert all((p == 0).all() for p in bridge.parameters())

    bridge.load_state_dict(sd, strict=True)

    # Compare against the snapshot directly rather than asserting "not all
    # zero" - LayerNorm bias legitimately initializes to all-zero, so that
    # check would pass even for a param that never got reloaded.
    reloaded = bridge.state_dict()
    for key, value in sd.items():
        assert torch.equal(reloaded[key], value), f"{key} did not round-trip"


def test_native_strict_true_raises_on_missing_key():
    bridge = TransformerBridge.boot_native(_native_cfg())
    sd = bridge.state_dict()
    incomplete = dict(sd)
    incomplete.pop(next(iter(incomplete)))

    with pytest.raises(RuntimeError, match="Missing key"):
        bridge.load_state_dict(incomplete, strict=True)


def test_native_strict_true_raises_on_unexpected_key():
    bridge = TransformerBridge.boot_native(_native_cfg())
    sd = dict(bridge.state_dict())
    sd["totally.bogus.key"] = torch.zeros(1)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        bridge.load_state_dict(sd, strict=True)


def test_native_strict_false_does_not_raise_on_partial_dict():
    bridge = TransformerBridge.boot_native(_native_cfg())
    sd = bridge.state_dict()
    first_key = next(iter(sd))
    partial = {first_key: sd[first_key]}

    result = bridge.load_state_dict(partial, strict=False)
    assert result.unexpected_keys == []
    assert len(result.missing_keys) > 0


def test_native_raw_keys_still_load_tracr_style():
    """boot_native's own raw parameter names must keep loading directly,
    mirroring tracr's make_tracr_transformer_bridge_state_dict compatibility
    contract (utilities/tracr.py)."""
    bridge = TransformerBridge.boot_native(_native_cfg())
    raw_sd = {k: v.clone() for k, v in bridge.original_model.state_dict().items()}

    with torch.no_grad():
        for p in bridge.parameters():
            p.zero_()

    bridge.load_state_dict(raw_sd, strict=True)

    reloaded_raw = bridge.original_model.state_dict()
    for key, value in raw_sd.items():
        assert torch.equal(reloaded_raw[key], value), f"{key} did not round-trip"


@pytest.mark.slow
def test_boot_transformers_round_trip_matches_forward_pass():
    """GPT-2's Conv1D-combined attention makes the bridge's q/k/v components
    storage-sharing VIEWS into c_attn, not independent parameters - so this is
    the case that actually exercises convert_hf_key_to_tl_key's HF-name
    renaming, not just identity passthrough like boot_native does."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.eval()

    torch.manual_seed(0)
    tokens = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        logits_before = bridge(tokens).clone()

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}

    with torch.no_grad():
        for p in bridge.parameters():
            p.zero_()

    bridge.load_state_dict(sd, strict=True)

    with torch.no_grad():
        logits_after = bridge(tokens).clone()

    max_diff = (logits_before - logits_after).abs().max().item()
    assert torch.allclose(
        logits_before, logits_after, atol=1e-5
    ), f"round trip did not restore forward-pass output: max diff={max_diff:.3e}"
