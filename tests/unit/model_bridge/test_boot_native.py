"""Tests for ``TransformerBridge.boot_native`` classmethod."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import LinearBridge
from transformer_lens.model_bridge.sources.native import NativeModel


def _cfg(**overrides) -> TransformerBridgeConfig:
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
        seed=0,
    )
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def test_native_adapter_weight_processing_conversions_shape():
    """Assert weight_processing_conversions is a dict whose keys reference real bridge slots; deliberately does not assert emptiness so added conversions don't force a test rewrite."""
    cfg = _cfg()
    bridge = TransformerBridge.boot_native(cfg)
    conversions = bridge.adapter.weight_processing_conversions
    # Must be a dict (base class allows None; native opts in).
    assert isinstance(conversions, dict), type(conversions).__name__
    # Every conversion key must reference a real bridge component root.
    for tl_path in conversions:
        root = tl_path.split(".")[0]
        assert hasattr(bridge, root), (
            f"weight_processing_conversions key {tl_path!r} references unknown "
            f"bridge root {root!r}"
        )


def test_native_block_forward_returns_single_element_tuple():
    """NativeBlock returns ``(hidden_states,)`` rather than a bare tensor to
    satisfy BlockBridge's HF-style output parser (block.py:227-240 expects a
    tuple whose first element is the residual stream). If BlockBridge evolves
    or NativeBlock is refactored to return a bare tensor, the failure mode is
    a confusing unpack error deep in block forward; pin the contract here."""
    from transformer_lens.model_bridge.sources.native.model import NativeBlock

    cfg = _cfg(n_layers=1)
    # NativeBlock's __init__ doesn't trigger NativeModel's d_mlp resolution;
    # set d_mlp explicitly so NativeMLP has a width to use.
    cfg.d_mlp = 4 * cfg.d_model
    block = NativeBlock(cfg)

    hidden = torch.randn(2, cfg.n_ctx, cfg.d_model)
    out = block(hidden)

    assert isinstance(out, tuple), f"NativeBlock must return tuple, got {type(out).__name__}"
    assert len(out) == 1, f"NativeBlock must return 1-tuple, got len={len(out)}"
    assert out[0].shape == hidden.shape


def test_boot_native_returns_bridge_over_native_model():
    bridge = TransformerBridge.boot_native(_cfg())
    assert isinstance(bridge, TransformerBridge)
    assert isinstance(bridge.original_model, NativeModel)


@pytest.mark.parametrize("stop_at_layer", [0, 2, -1])
def test_boot_native_direct_stop_matches_cached_stop(stop_at_layer: int):
    bridge = TransformerBridge.boot_native(_cfg(n_layers=3))
    bridge.eval()
    tokens = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        expected, _ = bridge.run_with_cache(tokens, stop_at_layer=stop_at_layer)
        actual = bridge(tokens, stop_at_layer=stop_at_layer)

    assert actual.shape == (1, 3, bridge.cfg.d_model)
    torch.testing.assert_close(actual, expected)


def test_native_state_dict_round_trip_restores_parameters():
    bridge = TransformerBridge.boot_native(_cfg())

    saved_state_dict = {key: value.detach().clone() for key, value in bridge.state_dict().items()}
    original_parameters = {
        name: parameter.detach().clone() for name, parameter in bridge.named_parameters()
    }

    with torch.no_grad():
        for parameter in bridge.parameters():
            parameter.zero_()

    result = bridge.load_state_dict(saved_state_dict, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []

    for name, parameter in bridge.named_parameters():
        torch.testing.assert_close(parameter, original_parameters[name])


def test_native_state_dict_strict_rejects_unexpected_keys():
    bridge = TransformerBridge.boot_native(_cfg())

    with pytest.raises(RuntimeError, match="Unexpected key"):
        bridge.load_state_dict(
            {"not.a.real.weight": torch.zeros(1)},
            strict=True,
        )


def test_boot_native_accepts_dict_config():
    cfg_dict = dict(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
    )
    bridge = TransformerBridge.boot_native(cfg_dict)
    assert bridge.cfg.d_model == 32
    assert bridge.cfg.architecture == "TransformerLensNative"


def test_boot_native_rejects_legacy_config_with_actionable_error():
    import pytest

    from transformer_lens import HookedTransformerConfig

    legacy_config = HookedTransformerConfig(
        n_layers=1,
        d_model=32,
        n_ctx=8,
        d_head=16,
        n_heads=2,
        d_vocab=16,
        act_fn="gelu",
    )

    with pytest.raises(
        TypeError,
        match=(
            "boot_native expected a TransformerBridgeConfig or dict, " "got HookedTransformerConfig"
        ),
    ):
        TransformerBridge.boot_native(legacy_config)


def test_boot_native_does_not_perturb_global_rng():
    """``boot_native(seed=...)`` must use a scoped torch.Generator instead of
    ``torch.manual_seed``. Otherwise a user calling boot_native then
    ``torch.randn(...)`` for batch sampling silently gets a deterministic
    sequence they didn't ask for."""
    # Snapshot what torch.randn(5) would produce starting from global seed 0.
    torch.manual_seed(0)
    expected_after = torch.randn(5)

    # Now re-seed globally to 0, build a seeded bridge, and confirm the next
    # torch.randn(5) still matches the pre-bridge prediction.
    torch.manual_seed(0)
    TransformerBridge.boot_native(_cfg(seed=42))
    actual_after = torch.randn(5)

    assert torch.equal(actual_after, expected_after), (
        "boot_native perturbed the global RNG — the next torch.randn diverged "
        f"from the pre-call baseline.\n  expected: {expected_after}\n  got:      {actual_after}"
    )


def test_boot_native_seed_is_honored():
    a = TransformerBridge.boot_native(_cfg(seed=123))
    b = TransformerBridge.boot_native(_cfg(seed=123))
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.allclose(pa, pb), f"Seed mismatch on {na}"


def test_boot_native_distinct_seeds_diverge():
    a = TransformerBridge.boot_native(_cfg(seed=1))
    b = TransformerBridge.boot_native(_cfg(seed=2))
    diffs = [
        not torch.allclose(pa, pb)
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters())
    ]
    assert any(diffs), "Two different seeds produced identical params"


def test_boot_native_skips_custom_init_when_disabled(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        pytest.fail("initialize_native_model was called with init_weights=False")

    def fail_if_forked(*_args, **_kwargs):
        pytest.fail("fork_rng was called with init_weights=False")

    monkeypatch.setattr(
        "transformer_lens.model_bridge.sources.native.initialize_native_model",
        fail_if_called,
    )
    monkeypatch.setattr(torch.random, "fork_rng", fail_if_forked)
    bridge = TransformerBridge.boot_native(_cfg(init_weights=False))

    assert isinstance(bridge.original_model, NativeModel)
    assert torch.count_nonzero(bridge.original_model.layers[0].attn.q.bias) > 0


def test_native_bridge_init_weights_reinitializes_in_place_and_honors_seed():
    bridge = TransformerBridge.boot_native(_cfg(seed=123))
    model = bridge.original_model
    expected = {name: param.detach().clone() for name, param in model.named_parameters()}

    with torch.no_grad():
        for param in model.parameters():
            param.fill_(42)

    bridge.init_weights()

    assert bridge.original_model is model
    for name, param in model.named_parameters():
        assert torch.equal(param, expected[name]), f"Seed mismatch on {name}"


def test_native_bridge_init_weights_does_not_perturb_global_rng():
    bridge = TransformerBridge.boot_native(_cfg(seed=42))
    torch.manual_seed(0)
    expected_after = torch.randn(5)

    torch.manual_seed(0)
    bridge.init_weights()
    actual_after = torch.randn(5)

    assert torch.equal(actual_after, expected_after)


def test_init_weights_rejects_non_native_bridge():
    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

    class StubAdapter(ArchitectureAdapter):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.component_mapping = {"stub_proj": LinearBridge(name="proj")}

    cfg = _cfg(architecture="StubForTest")
    bridge = TransformerBridge(StubModel(), StubAdapter(cfg), tokenizer=None)

    with pytest.raises(RuntimeError, match=r"boot_native.*StubModel"):
        bridge.init_weights()


def test_boot_native_forward_and_cache():
    cfg = _cfg()
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    logits = bridge(inputs, return_type="logits")
    assert logits.shape == (2, cfg.n_ctx, cfg.d_vocab)
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    assert "blocks.0.attn.hook_pattern" in cache


@pytest.mark.parametrize("prepend_bos", [True, False])
def test_run_with_cache_forwards_prepend_bos_for_string_input(monkeypatch, prepend_bos):
    cfg = _cfg()
    bridge = TransformerBridge.boot_native(cfg)
    bridge._tokenizer = object()
    tokenization_calls = []

    def to_tokens(input, prepend_bos=None, padding_side=None):
        tokenization_calls.append((input, prepend_bos, padding_side))
        if prepend_bos is None:
            prepend_bos = bridge.cfg.default_prepend_bos
        tokens = [0, 7] if prepend_bos else [7]
        return torch.tensor([tokens])

    monkeypatch.setattr(bridge, "to_tokens", to_tokens)
    bridge.eval()

    with torch.no_grad():
        direct_logits = bridge("hello", prepend_bos=prepend_bos)
        cached_logits, cache = bridge.run_with_cache("hello", prepend_bos=prepend_bos)

    assert tokenization_calls == [
        ("hello", prepend_bos, None),
        ("hello", prepend_bos, None),
    ]
    torch.testing.assert_close(cached_logits, direct_logits)
    assert cache["hook_embed"].shape[1] == direct_logits.shape[1]


def test_boot_native_does_not_load_transformers_runtime():
    # Sanity that the native path doesn't depend on HuggingFace's `transformers`
    # for the runtime work — we check that calling boot_native doesn't trigger
    # an AutoModel/AutoTokenizer import. (`transformers` is in the dependency
    # set, but the native code path should not touch it.)
    sys.modules.pop("transformers.models.auto", None)
    TransformerBridge.boot_native(_cfg())
    # If boot_native loaded an HF auto class, `transformers.models.auto` would
    # be in sys.modules. Not bullet-proof (other paths may import it earlier in
    # the same process) but catches accidental coupling in isolation.


def test_native_adapter_rejects_colliding_attribute_names():
    """If a module ever exposes ``embed`` / ``blocks`` / etc. as top-level
    attributes, bridge construction would die in ``add_module`` with an opaque
    KeyError. The adapter should reject it at prepare_model time with a
    diagnostic pointing at the real cause."""
    import pytest
    import torch.nn as nn

    from transformer_lens.model_bridge.sources import build_bridge_from_module

    class CollidingModel(nn.Module):
        def __init__(self):
            super().__init__()
            # "embed" collides with the bridge's component slot.
            self.embed = nn.Embedding(8, 4)
            self.layers = nn.ModuleList()

        def forward(self, input_ids):
            return self.embed(input_ids)

    with pytest.raises(ValueError, match="collide with bridge component slots"):
        build_bridge_from_module(
            CollidingModel(),
            architecture="TransformerLensNative",
            tl_config=_cfg(),
        )


def test_boot_native_rejects_foreign_architecture_string():
    """If config.architecture names a real-model adapter (e.g. copied from a
    Llama config), boot_native would dispatch to that adapter and fail opaquely
    in prepare_model. Refuse it explicitly with a pointing diagnostic."""
    import pytest

    cfg = _cfg()
    cfg.architecture = "LlamaForCausalLM"
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        TransformerBridge.boot_native(cfg)

    # Explicit "TransformerLensNative" is allowed (it's the value boot_native
    # would default to anyway).
    cfg2 = _cfg()
    cfg2.architecture = "TransformerLensNative"
    bridge = TransformerBridge.boot_native(cfg2)
    assert bridge.cfg.architecture == "TransformerLensNative"


def test_native_adapter_rejects_non_submodule_collisions():
    """The bridge's ``__getattr__`` fallback finds *any* attribute on the
    underlying model — buffers, plain tensors, properties — not just
    registered submodules. Each of these must also be caught at prepare_model
    time. Without this, a model with ``self.unembed = torch.zeros(...)`` (a
    buffer or plain attribute) would silently break add_module at bridge setup.
    """
    import pytest
    import torch.nn as nn

    from transformer_lens.model_bridge.sources import build_bridge_from_module

    class BufferCollidesModel(nn.Module):
        """Registers ``unembed`` as a buffer — not a submodule, but still
        visible via ``getattr``."""

        def __init__(self):
            super().__init__()
            self.tok_embed = nn.Embedding(8, 4)
            self.register_buffer("unembed", torch.zeros(4, 8))

        def forward(self, input_ids):
            return self.tok_embed(input_ids) @ self.unembed

    with pytest.raises(ValueError, match=r"\['unembed'\]"):
        build_bridge_from_module(
            BufferCollidesModel(),
            architecture="TransformerLensNative",
            tl_config=_cfg(),
        )

    class PropertyCollidesModel(nn.Module):
        """Exposes ``blocks`` as a property — neither a submodule nor a buffer,
        but a __getattr__ fallback would still resolve it."""

        def __init__(self):
            super().__init__()
            self.tok_embed = nn.Embedding(8, 4)

        @property
        def blocks(self):
            return []

        def forward(self, input_ids):
            return self.tok_embed(input_ids)

    with pytest.raises(ValueError, match=r"\['blocks'\]"):
        build_bridge_from_module(
            PropertyCollidesModel(),
            architecture="TransformerLensNative",
            tl_config=_cfg(),
        )


def test_boot_native_resolves_d_mlp_default():
    """If the caller didn't pin d_mlp, the bridge's cfg must report the
    resolved value (4 * d_model) instead of None. NativeMLP independently
    falling back to 4 * d_model is wrong: downstream consumers (telemetry,
    save/load, demo notebooks) need cfg.d_mlp to reflect what the model built."""
    # Build a config with d_mlp explicitly None to force the default path.
    cfg_dict = dict(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        act_fn="gelu",
        normalization_type="LN",
    )
    bridge = TransformerBridge.boot_native(cfg_dict)
    assert bridge.cfg.d_mlp == 4 * bridge.cfg.d_model

    # And the underlying MLP's actual hidden width must match.
    mlp = bridge.original_model.layers[0].mlp
    assert mlp.fc_in.out_features == bridge.cfg.d_mlp


def test_boot_native_does_not_mutate_supplied_config():
    """boot_native sets a default architecture when missing — but it must do
    that on a local copy, not on the caller's config object. Same hazard as
    build_bridge_from_module."""
    cfg = _cfg()
    assert cfg.architecture is None  # baseline: no architecture set

    snapshot = {k: getattr(cfg, k) for k in ("architecture", "model_name", "dtype", "device")}
    TransformerBridge.boot_native(cfg)
    for field, before in snapshot.items():
        after = getattr(cfg, field)
        assert before == after, f"boot_native mutated cfg.{field}: {before!r} -> {after!r}"


def test_native_gelu_new_uses_tanh_approximation():
    """gelu_new must compute the tanh-approximation that HF GPT-2 and
    HookedTransformer use, not plain (erf-based) GELU. A plain alias would
    produce small but persistent drift in parity comparisons."""
    import torch.nn.functional as F

    from transformer_lens.model_bridge.sources.native.model import _ACTIVATIONS

    x = torch.linspace(-3.0, 3.0, 64)
    gelu_new_out = _ACTIVATIONS["gelu_new"](x)
    plain_gelu_out = _ACTIVATIONS["gelu"](x)
    tanh_ref = F.gelu(x, approximate="tanh")

    # Exact match to the tanh-approximation formula.
    assert torch.allclose(gelu_new_out, tanh_ref)
    # And distinguishable from plain erf-based GELU.
    assert not torch.allclose(gelu_new_out, plain_gelu_out, atol=1e-5)


def test_boot_native_supports_training_step():
    """Regression for #1324 — backward hooks must clean up so .backward()
    produces real gradients on bridge params during training."""
    cfg = _cfg(n_layers=2)
    bridge = TransformerBridge.boot_native(cfg)
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)

    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    loss = bridge(inputs, return_type="loss")
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.parameters()
    ), "No non-zero gradients after backward"
    optimizer.step()
    optimizer.zero_grad()


def test_boot_native_resolves_initializer_range_sentinel():
    """Regression for #1568 — the -1.0 sentinel must be resolved on the
    bridge config path the same way HookedTransformerConfig resolves it
    (hooked_transformer_config.py), instead of silently falling back to the
    unrelated std=0.02 default inside native/init.py."""
    import math

    cfg = _cfg(init_mode="gpt2")
    # _cfg() doesn't set initializer_range, so it should still carry the
    # -1.0 sentinel until __post_init__ resolves it.
    expected = 0.8 / math.sqrt(cfg.d_model)
    assert cfg.initializer_range == pytest.approx(expected)


def test_boot_native_kaiming_gain_scales_weights():
    """Regression for #1568 — xavier/kaiming init must use initializer_range
    as a multiplicative gain. Without this, the config value is silently
    ignored and every kaiming/xavier model gets the same fixed scale
    regardless of what the caller asked for."""
    cfg_gain_1 = _cfg(init_mode="kaiming_normal", initializer_range=1.0, seed=0)
    cfg_gain_2 = _cfg(init_mode="kaiming_normal", initializer_range=2.0, seed=0)

    bridge_1 = TransformerBridge.boot_native(cfg_gain_1)
    bridge_2 = TransformerBridge.boot_native(cfg_gain_2)

    std_1 = bridge_1.W_E.std().item()
    std_2 = bridge_2.W_E.std().item()

    # Same seed, only gain differs -> std should scale ~proportionally.
    assert std_2 / std_1 == pytest.approx(2.0, rel=0.15)
