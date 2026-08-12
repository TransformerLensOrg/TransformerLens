"""Regression tests for TransformerBridge.state_dict()/load_state_dict() round-tripping (#1587).

state_dict() emits TL-renamed keys (e.g. "blocks.0.attn.q.weight"), but
load_state_dict() only matched raw native parameter names, so a
state_dict() -> load_state_dict() round trip silently loaded nothing and
strict=True was silently downgraded to strict=False.

Also covers "copy-split staleness" (#1637): load_state_dict(..., assign=True)
used to replace view-backed split-component parameters (e.g. gpt2's q/k/v,
which are torch.tensor_split views into the combined c_attn weight) wholesale,
desyncing them from the combined weight they share storage with.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.transformer_bridge import _is_view_backed


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


def test_native_clean_key_dict_with_partial_aliases_does_not_raise_strict():
    """A complete raw-HF-format-style state dict (clean keys, _original_component
    stripped) writes only one alias per shared-storage TL key -- boot_native's own
    wrapping produces this aliasing internally (e.g. "layers.0.ln1.weight" is
    reachable via two different _original_component paths onto the same
    Parameter), not just gpt2's c_attn split. Since aliases of the same tensor
    share storage, writing any one of them is sufficient; strict=True must not
    report the other, unwritten aliases as missing."""
    bridge = TransformerBridge.boot_native(_native_cfg())

    raw_sd = bridge.original_model.state_dict()
    clean_to_actuals: dict[str, list[str]] = {}
    for actual_key in raw_sd:
        if actual_key != "_original_component":
            clean_to_actuals.setdefault(actual_key.replace("._original_component", ""), []).append(
                actual_key
            )
    assert any(len(keys) > 1 for keys in clean_to_actuals.values()), (
        "fixture assumption broken: expected boot_native to have some "
        "clean key reachable through more than one actual path"
    )

    # One representative actual key's value per clean key, same shape as a
    # real raw-HF-format checkpoint (no duplicate paths for the same param).
    clean_sd = {clean_key: raw_sd[keys[0]].clone() for clean_key, keys in clean_to_actuals.items()}

    with torch.no_grad():
        for p in bridge.parameters():
            p.zero_()

    result = bridge.load_state_dict(clean_sd, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    reloaded_raw = bridge.original_model.state_dict()
    for clean_key, actual_keys in clean_to_actuals.items():
        value = clean_sd[clean_key]
        for actual_key in actual_keys:
            assert torch.equal(
                reloaded_raw[actual_key], value
            ), f"{actual_key} (alias of {clean_key}) did not round-trip"


def test_is_view_backed_detects_tensor_split_views():
    """Core detection helper for #1637: a torch.tensor_split view shares storage
    with a larger source tensor, even after nn.Parameter wrapping (which does not
    reliably preserve Tensor._base view tracking, so this must not rely on it)."""
    combined = torch.nn.Parameter(torch.randn(12, 4))
    a, b, c = torch.tensor_split(combined, 3, dim=0)
    split_param = torch.nn.Parameter(a)
    assert split_param.untyped_storage().data_ptr() == combined.untyped_storage().data_ptr()
    assert _is_view_backed(split_param)

    assert not _is_view_backed(combined)
    assert not _is_view_backed(torch.nn.Parameter(torch.randn(4, 4)))


def test_native_assign_true_round_trip_no_split_components():
    """boot_native's components are independent parameters (no split/view
    components), so assign=True should take the ordinary passthrough path and
    round-trip exactly like assign=False does."""
    bridge = TransformerBridge.boot_native(_native_cfg())

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    with torch.no_grad():
        for p in bridge.parameters():
            p.zero_()

    bridge.load_state_dict(sd, strict=True, assign=True)

    reloaded = bridge.state_dict()
    for key, value in sd.items():
        assert torch.equal(reloaded[key], value), f"{key} did not round-trip under assign=True"


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


@pytest.mark.slow
def test_boot_transformers_clean_key_dict_does_not_raise_strict():
    """Reported review case on real gpt2: a complete raw-HF-format-style state
    dict (clean keys) writes only one alias per shared-storage TL key, since
    gpt2's split q/k/v are views into c_attn reachable via multiple actual
    paths. strict=True previously raised ~337 false "missing key" errors even
    though the load fully restores the forward pass."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

    raw_sd = bridge.original_model.state_dict()
    clean_sd = {
        actual_key.replace("._original_component", ""): value.clone()
        for actual_key, value in raw_sd.items()
        if actual_key != "_original_component"
    }
    assert len(clean_sd) < len(raw_sd), "fixture assumption broken: expected some aliasing on gpt2"

    result = bridge.load_state_dict(clean_sd, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.slow
def test_boot_transformers_assign_true_does_not_leave_combined_weight_stale():
    """#1637 repro: assign=True on a split QKV component used to replace the
    parameter object instead of copying into it, breaking the view relationship
    with c_attn -- the bridge itself read the new value, but
    original_model.state_dict() (what save_pretrained() exports) silently kept
    the pre-load data for the combined weight."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    mutated = dict(sd)
    mutated["blocks.0.attn.q.weight"] = sd["blocks.0.attn.q.weight"] + 100.0

    bridge.load_state_dict(mutated, strict=True, assign=True)

    assert torch.equal(
        bridge.blocks[0].attn.q.original_component.weight, mutated["blocks.0.attn.q.weight"]
    )

    raw_sd = bridge.original_model.state_dict()
    c_attn_w = raw_sd["transformer.h.0.attn._original_component.c_attn._original_component.weight"]
    d_model = bridge.cfg.d_model
    assert torch.allclose(c_attn_w[:, :d_model].T, mutated["blocks.0.attn.q.weight"])


@pytest.mark.slow
def test_boot_transformers_assign_true_shape_mismatch_raises_clear_error():
    """A view-backed split component can only be loaded under assign=True via an
    in-place copy, which requires a matching shape -- fail loudly instead of a
    confusing error surfacing from deep inside copy_, or silently corrupting data."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

    sd = dict(bridge.state_dict())
    sd["blocks.0.attn.q.weight"] = sd["blocks.0.attn.q.weight"][:-1]

    with pytest.raises(RuntimeError, match="view sharing storage"):
        bridge.load_state_dict(sd, strict=True, assign=True)
