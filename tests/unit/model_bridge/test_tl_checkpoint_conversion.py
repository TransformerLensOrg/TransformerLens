"""Tests for the legacy TL-property-format checkpoint converter (#1588).

Historical HookedTransformer checkpoints (OthelloGPT, grokking, ARENA content)
are saved with the old property-style keys ("blocks.0.attn.W_Q", "embed.W_E",
...) and per-head tensor shapes. `convert_tl_checkpoint` maps those onto the
key/tensor format `TransformerBridge.boot_native(cfg).load_state_dict` accepts
natively, so these checkpoints can be loaded once and re-saved in bridge
format without teaching `load_state_dict` a second key convention.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.config import HookedTransformerConfig, TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.tl_checkpoint_conversion import convert_tl_checkpoint


def _cfg_kwargs(**overrides):
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
    return base


def _ht_and_bridge_cfg(**overrides):
    kwargs = _cfg_kwargs(**overrides)
    return HookedTransformerConfig(**kwargs), TransformerBridgeConfig(**kwargs)


def test_convert_tl_checkpoint_loads_strict_into_native_bridge():
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg()
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)

    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_convert_tl_checkpoint_matches_source_forward_pass():
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg()
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    bridge.load_state_dict(converted, strict=True)

    tokens = torch.randint(0, ht_cfg.d_vocab, (1, 4))
    with torch.no_grad():
        ht_logits = ht(tokens)
        bridge_logits = bridge(tokens)

    torch.testing.assert_close(bridge_logits, ht_logits, atol=1e-4, rtol=1e-4)


def test_convert_tl_checkpoint_places_qkvo_in_correct_head_slots():
    """Independent check that per-head Q/K/V/O land in the right slots: read
    the converted+loaded bridge back out through its own W_Q/W_K/W_V/W_O
    properties (implemented separately from the converter) and compare
    directly against the source HookedTransformer's per-head weights, rather
    than trusting the converter's own reshape math."""
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg()
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    bridge.load_state_dict(converted, strict=True)

    torch.testing.assert_close(bridge.W_Q, ht.W_Q)
    torch.testing.assert_close(bridge.W_K, ht.W_K)
    torch.testing.assert_close(bridge.W_V, ht.W_V)
    torch.testing.assert_close(bridge.W_O, ht.W_O)
    torch.testing.assert_close(bridge.b_Q, ht.b_Q)
    torch.testing.assert_close(bridge.b_K, ht.b_K)
    torch.testing.assert_close(bridge.b_V, ht.b_V)
    torch.testing.assert_close(bridge.b_O, ht.b_O)


def test_convert_tl_checkpoint_raises_on_cfg_mismatch():
    """A wrong cfg can't be caught by a downstream shape-mismatch error --
    merging per-head dims produces a validly-shaped result for any head
    count, since d_model == n_heads * d_head for any factoring of it. The
    converter must catch this itself."""
    ht_cfg, _ = _ht_and_bridge_cfg()
    ht = HookedTransformer(ht_cfg)

    wrong_cfg = TransformerBridgeConfig(
        **_cfg_kwargs(n_heads=4, d_head=8)
    )  # same d_model, wrong split

    with pytest.raises(ValueError, match="attn.W_Q"):
        convert_tl_checkpoint(ht.state_dict(), wrong_cfg)


def test_convert_tl_checkpoint_raises_on_unrecognized_key():
    _, bridge_cfg = _ht_and_bridge_cfg()
    with pytest.raises(ValueError, match="not a recognized"):
        convert_tl_checkpoint({"blocks.0.attn.totally_unknown_param": torch.zeros(1)}, bridge_cfg)


def test_convert_tl_checkpoint_supports_gqa():
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg(n_heads=4, d_head=8, n_key_value_heads=2)
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []

    tokens = torch.randint(0, ht_cfg.d_vocab, (1, 4))
    with torch.no_grad():
        ht_logits = ht(tokens)
        bridge_logits = bridge(tokens)
    torch.testing.assert_close(bridge_logits, ht_logits, atol=1e-4, rtol=1e-4)


def test_convert_tl_checkpoint_supports_lnpre():
    """OthelloGPT (this converter's motivating use case, #1588) uses
    normalization_type="LNPre" -- param-free pre-norm, so ln1/ln2/ln_final
    have no weight/bias keys at all in the state dict for this converter to
    handle; this just confirms the round trip still works end to end."""
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg(normalization_type="LNPre")
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []

    tokens = torch.randint(0, ht_cfg.d_vocab, (1, 4))
    with torch.no_grad():
        ht_logits = ht(tokens)
        bridge_logits = bridge(tokens)
    torch.testing.assert_close(bridge_logits, ht_logits, atol=1e-4, rtol=1e-4)


def test_convert_tl_checkpoint_supports_attn_only():
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg(attn_only=True)
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []

    tokens = torch.randint(0, ht_cfg.d_vocab, (1, 4))
    with torch.no_grad():
        ht_logits = ht(tokens)
        bridge_logits = bridge(tokens)
    torch.testing.assert_close(bridge_logits, ht_logits, atol=1e-4, rtol=1e-4)


def test_convert_tl_checkpoint_drops_attention_buffers():
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg()
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)

    assert not any(key.endswith((".mask", ".IGNORE")) for key in converted)


def test_convert_tl_checkpoint_supports_gated_mlp_and_rms_norm():
    """The native bridge's gated MLP has no bias parameter at all for
    gate/in/out (matching how real gated-MLP HF architectures like Llama are
    built) while HookedTransformer's gated MLP keeps live b_in/b_out
    parameters (pre-existing mismatch between the two implementations,
    unrelated to this converter). convert_tl_checkpoint still faithfully
    translates those keys since they're real HT parameters; load_state_dict
    is the one that should refuse them under strict=True. Here they're
    exactly zero (freshly constructed, untrained model) so dropping them via
    strict=False is lossless and the forward pass still matches exactly.
    """
    ht_cfg, bridge_cfg = _ht_and_bridge_cfg(gated_mlp=True, normalization_type="RMS", act_fn="silu")
    ht = HookedTransformer(ht_cfg)

    converted = convert_tl_checkpoint(ht.state_dict(), bridge_cfg)
    bridge = TransformerBridge.boot_native(bridge_cfg)
    result = bridge.load_state_dict(converted, strict=False)

    assert result.missing_keys == []
    assert set(result.unexpected_keys) == {
        f"blocks.{i}.mlp.{part}.bias" for i in range(ht_cfg.n_layers) for part in ("in", "out")
    }

    tokens = torch.randint(0, ht_cfg.d_vocab, (1, 4))
    with torch.no_grad():
        ht_logits = ht(tokens)
        bridge_logits = bridge(tokens)
    torch.testing.assert_close(bridge_logits, ht_logits, atol=1e-4, rtol=1e-4)
