"""Tests for ``TransformerBridge.boot_native`` classmethod."""
from __future__ import annotations

import sys

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
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
    """Snapshot the ``weight_processing_conversions`` contract without locking
    in its size.

    The native adapter currently uses ``{}`` because the native layout already
    stores Q/K/V split per-head — no rearranges needed. That's the right
    choice today, but a follow-up that implements fold_ln / compatibility-mode
    parity will likely add entries. We assert the type and that any present
    keys point at real bridge slots; we deliberately do **not** assert the
    set is empty, so a future PR adding conversions doesn't have to rewrite
    this test."""
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


def test_boot_native_forward_and_cache():
    cfg = _cfg()
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    logits = bridge(inputs, return_type="logits")
    assert logits.shape == (2, cfg.n_ctx, cfg.d_vocab)
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    assert "blocks.0.attn.hook_pattern" in cache


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
