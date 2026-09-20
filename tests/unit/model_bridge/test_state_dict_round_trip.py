"""Regression tests for TransformerBridge.state_dict()/load_state_dict() round-tripping (#1587).

state_dict() emits TL-renamed keys (e.g. "blocks.0.attn.q.weight"), but
load_state_dict() only matched raw native parameter names, so a
state_dict() -> load_state_dict() round trip silently loaded nothing and
strict=True was silently downgraded to strict=False.

Also covers "copy-split staleness" (#1637): load_state_dict(..., assign=True)
used to replace parameters that share underlying storage with something else
(a split QKV/gate-up component's view into its combined weight, the combined
weight itself, or a tied pair like embed/unembed, #1725) wholesale, desyncing
them from whatever they share storage with.
"""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Phi3Config, Phi3ForCausalLM

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.model_bridge.transformer_bridge import _storage_group_keys


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


def _tiny_gpt2_bridge() -> TransformerBridge:
    """Real, randomly-initialized tiny GPT-2 bridge (JointQKVAttentionBridge path).

    Built from a config, not a download -- fast enough for the default test
    tier, unlike the boot_transformers("gpt2") tests below.
    """
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
    torch.manual_seed(0)
    return build_bridge_from_module(
        GPT2LMHeadModel(cfg), architecture="GPT2LMHeadModel", hf_config=cfg
    )


def _tiny_phi3_bridge() -> TransformerBridge:
    """Real, randomly-initialized tiny Phi-3 bridge (JointGateUpMLPBridge path)."""
    cfg = Phi3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    return build_bridge_from_module(
        Phi3ForCausalLM(cfg), architecture="Phi3ForCausalLM", hf_config=cfg
    )


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


def test_storage_group_keys_detects_shared_storage():
    """Core detection helper for #1637/#1725: any key whose tensor shares
    underlying storage with another key's belongs to the same group, whether
    it's the smaller view, the larger owner, or a same-size tied duplicate --
    even though nn.Parameter wrapping doesn't reliably preserve Tensor._base,
    so this can't rely on that."""
    combined = torch.nn.Parameter(torch.randn(12, 4))
    view = torch.nn.Parameter(torch.tensor_split(combined, 3, dim=0)[0])
    tied_a = torch.nn.Parameter(torch.randn(4, 4))
    tied_b = torch.nn.Parameter(tied_a.data)
    independent = torch.nn.Parameter(torch.randn(4, 4))

    state_dict = {
        "combined": combined,
        "view": view,
        "tied_a": tied_a,
        "tied_b": tied_b,
        "independent": independent,
    }
    grouped = _storage_group_keys(state_dict)
    assert grouped == {"combined", "view", "tied_a", "tied_b"}


def test_storage_group_keys_excludes_meta_tensors():
    """Every meta tensor reports untyped_storage().data_ptr() == 0 (no real
    backing memory), so comparing meta tensors by data pointer would
    spuriously group every unrelated offloaded parameter in the model
    together. Meta-target keys must never be flagged as a storage group."""
    a = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
    b = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() == 0

    grouped = _storage_group_keys({"a": a, "b": b})
    assert grouped == set()


def test_native_assign_true_round_trip_no_split_components():
    """boot_native's components are independent parameters (no split/view or
    tied components), so assign=True should take the ordinary passthrough
    path and round-trip exactly like assign=False does -- and, unlike
    assign=False, actually adopt the incoming tensor objects rather than
    copying into the existing ones, since that's the point of assign=True."""
    bridge = TransformerBridge.boot_native(_native_cfg())

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    with torch.no_grad():
        for p in bridge.parameters():
            p.zero_()

    bridge.load_state_dict(sd, strict=True, assign=True)

    reloaded = bridge.state_dict()
    for key, value in sd.items():
        assert torch.equal(reloaded[key], value), f"{key} did not round-trip under assign=True"

    first_key = next(iter(sd))
    assert (
        bridge.state_dict()[first_key].data_ptr() == sd[first_key].data_ptr()
    ), "non-shared key should be assigned (adopt the incoming tensor object), not copied into"


def test_tiny_gpt2_assign_true_does_not_leave_combined_weight_stale():
    """Fast (no-download) CI-covered version of the #1637 repro on the QKV
    split path: assign=True on a split component must not desync it from the
    combined weight (c_attn) it shares storage with."""
    bridge = _tiny_gpt2_bridge()

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


def test_tiny_phi3_assign_true_does_not_leave_combined_weight_stale():
    """Fast (no-download) CI-covered version of the #1637 repro on the
    gate/up split path (JointGateUpMLPBridge)."""
    bridge = _tiny_phi3_bridge()

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    gate_key = next(k for k in sd if k.endswith("mlp.gate.weight"))
    mutated = dict(sd)
    mutated[gate_key] = sd[gate_key] + 100.0

    bridge.load_state_dict(mutated, strict=True, assign=True)

    gate_component = bridge
    for part in gate_key.split(".")[:-1]:
        gate_component = (
            getattr(gate_component, part) if not part.isdigit() else gate_component[int(part)]
        )
    assert torch.equal(gate_component.original_component.weight, mutated[gate_key])

    raw_sd = bridge.original_model.state_dict()
    combined_key = next(
        k
        for k in raw_sd
        if "gate_up_proj" in k and k.endswith("weight") and "0" in k.split(".")[:3]
    )
    d_mlp = bridge.cfg.d_mlp
    combined_w = raw_sd[combined_key]
    assert torch.allclose(combined_w[:d_mlp, :], mutated[gate_key])


def test_tiny_gpt2_assign_true_raw_combined_key_does_not_orphan_views():
    """Reviewer-flagged gap (jlarson4, PR #1660): loading the combined
    weight's own *raw* key directly (not through a split alias) under
    assign=True must not orphan the q/k/v views that share its storage --
    otherwise the bridge's own forward pass would keep reading stale q/k/v
    data while original_model.state_dict() shows the new c_attn value."""
    bridge = _tiny_gpt2_bridge()

    raw_sd = bridge.original_model.state_dict()
    c_attn_key = next(
        k for k in raw_sd if "h.0.attn" in k and k.endswith("c_attn._original_component.weight")
    )
    new_c_attn = raw_sd[c_attn_key].clone() + 100.0

    bridge.load_state_dict({c_attn_key: new_c_attn}, strict=False, assign=True)

    d_model = bridge.cfg.d_model
    expected_q = new_c_attn[:, :d_model].T
    assert torch.equal(
        bridge.blocks[0].attn.q.original_component.weight, expected_q
    ), "q view was orphaned by a direct assign=True write to the combined c_attn key"


def test_assign_true_dtype_mismatch_raises_clear_error():
    """Reviewer-flagged gap (jlarson4): only shape was guarded; a dtype
    mismatch on a storage-shared target would otherwise silently
    copy-convert instead of erroring, e.g. producing a silently mixed-dtype
    model from one call."""
    bridge = _tiny_gpt2_bridge()

    sd = dict(bridge.state_dict())
    sd["blocks.0.attn.q.weight"] = sd["blocks.0.attn.q.weight"].to(torch.float64)

    with pytest.raises(RuntimeError, match="dtype"):
        bridge.load_state_dict(sd, strict=True, assign=True)


def test_assign_true_device_mismatch_raises_clear_error():
    """Reviewer-flagged gap (jlarson4): a device mismatch on a storage-shared
    target must also be rejected explicitly, not silently accepted."""
    bridge = _tiny_gpt2_bridge()

    sd = dict(bridge.state_dict())
    sd["blocks.0.attn.q.weight"] = sd["blocks.0.attn.q.weight"].to("meta")

    with pytest.raises(RuntimeError, match="device"):
        bridge.load_state_dict(sd, strict=True, assign=True)


def test_assign_true_meta_target_materializes_via_passthrough():
    """koriyoshi2041's review finding (PR #1660): copy_() onto a meta target
    silently no-ops rather than raising, so a naive view-backed branch could
    report a successful load while the parameter stays meta forever. Meta
    targets are excluded from the storage-group check (see
    test_storage_group_keys_excludes_meta_tensors) and instead take the
    ordinary assign=True path, which correctly materializes them -- the
    standard way to load real weights onto a meta-initialized model."""
    bridge = _tiny_gpt2_bridge()

    current = bridge.original_model.state_dict(keep_vars=True)
    q_key = next(k for k in current if "h.0.attn.q" in k and k.endswith("weight"))
    q_param = current[q_key]
    assert not q_param.is_meta

    # A plain `.data = ...to("meta")` reassignment is rejected by PyTorch
    # (incompatible tensor type), so simulate an offloaded/meta parameter by
    # replacing the owning submodule's Parameter object outright, the same
    # way accelerate's own offload hooks do it.
    owner_path = ".".join(q_key.split(".")[:-1])
    owner = bridge.original_model.get_submodule(owner_path)
    owner.weight = torch.nn.Parameter(q_param.data.to("meta"), requires_grad=q_param.requires_grad)
    assert bridge.original_model.state_dict(keep_vars=True)[q_key].is_meta

    new_value = torch.randn(q_param.shape)
    bridge.load_state_dict({"blocks.0.attn.q.weight": new_value}, strict=False, assign=True)

    reloaded = bridge.original_model.state_dict(keep_vars=True)[q_key]
    assert not reloaded.is_meta, "meta target was not materialized by assign=True"
    assert torch.equal(reloaded, new_value)


def test_assign_true_validates_all_keys_before_copying_any():
    """Reviewer-flagged gap (jlarson4): the loop must validate every
    storage-shared key before applying any copy_, not copy-as-it-validates --
    otherwise a failure partway through leaves some keys already written and
    others untouched instead of the whole call atomically failing."""
    bridge = _tiny_gpt2_bridge()

    sd = dict(bridge.state_dict())
    original_k = sd["blocks.0.attn.k.weight"].clone()
    sd["blocks.0.attn.k.weight"] = sd["blocks.0.attn.k.weight"] + 100.0  # valid
    sd["blocks.0.attn.q.weight"] = sd["blocks.0.attn.q.weight"][:-1]  # shape mismatch

    with pytest.raises(RuntimeError):
        bridge.load_state_dict(sd, strict=True, assign=True)

    assert torch.equal(
        bridge.blocks[0].attn.k.original_component.weight, original_k
    ), "a valid key was written before the whole call raised on a different key's mismatch"


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
    """A storage-shared target can only be loaded under assign=True via an
    in-place copy, which requires a matching shape -- fail loudly instead of a
    confusing error surfacing from deep inside copy_, or silently corrupting data."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

    sd = dict(bridge.state_dict())
    sd["blocks.0.attn.q.weight"] = sd["blocks.0.attn.q.weight"][:-1]

    with pytest.raises(RuntimeError, match="shares storage"):
        bridge.load_state_dict(sd, strict=True, assign=True)


@pytest.mark.slow
def test_boot_transformers_assign_true_tied_embed_unembed_stays_in_sync():
    """#1725 repro, fixed as a consequence of the same storage-group
    generalization: gpt2 ties embed/unembed weights (separate nn.Parameter
    objects sharing storage, confirmed via data_ptr equality). Loading only
    embed.weight under assign=True must not leave unembed reading stale
    pre-load data -- previously this broke the bridge's own forward pass,
    not just the exported checkpoint."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.eval()
    tokens = torch.randint(0, 1000, (1, 6))
    with torch.no_grad():
        logits_before = bridge(tokens).clone()

    sd = {k: v.clone() for k, v in bridge.state_dict().items()}
    new_embed = sd["embed.weight"] + 3.0
    bridge.load_state_dict({"embed.weight": new_embed}, strict=False, assign=True)

    assert torch.equal(bridge.unembed.original_component.weight, new_embed)

    with torch.no_grad():
        logits_after = bridge(tokens).clone()
    assert not torch.allclose(
        logits_before, logits_after
    ), "loading a new embedding matrix should change the forward pass output"
