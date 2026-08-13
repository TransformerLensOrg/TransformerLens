"""Native init: seeded reproducibility across device/dtype, and initializer_range gain."""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources.native.init import initialize_native_model


def _cfg(**overrides):
    base = dict(
        n_layers=2,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=100,
        n_ctx=16,
        act_fn="gelu",
        seed=0,
    )
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def test_seeded_reinit_after_dtype_cast_reproduces_boot_weights():
    """Same seed must give the same weights whether init runs before or after
    the .to(dtype) cast (boot inits first; init_weights() runs after)."""
    cfg = _cfg()
    boot_then_cast = TransformerBridge.boot_native(cfg).to(torch.float16)

    recast = TransformerBridge.boot_native(cfg).to(torch.float16)
    native = recast.original_model
    initialize_native_model(native, cfg)  # re-init AFTER the cast

    for (name, a), (_, b) in zip(
        boot_then_cast.original_model.named_parameters(), native.named_parameters()
    ):
        assert torch.equal(a, b), f"{name} diverged between init-before and init-after cast"


def test_seeded_init_is_deterministic():
    cfg = _cfg()
    a = TransformerBridge.boot_native(cfg)
    b = TransformerBridge.boot_native(cfg)
    for (name, pa), (_, pb) in zip(
        a.original_model.named_parameters(), b.original_model.named_parameters()
    ):
        assert torch.equal(pa, pb), name


@pytest.mark.parametrize("mode", ["xavier_normal", "kaiming_normal"])
def test_initializer_range_scales_non_gpt2_modes(mode):
    """An explicit initializer_range acts as the gain (legacy semantics)."""
    plain = TransformerBridge.boot_native(_cfg(init_mode=mode))
    scaled = TransformerBridge.boot_native(_cfg(init_mode=mode, initializer_range=0.5))

    w_plain = plain.original_model.layers[0].attn.q.weight
    w_scaled = scaled.original_model.layers[0].attn.q.weight
    ratio = (w_scaled.std() / w_plain.std()).item()
    assert ratio == pytest.approx(0.5, rel=0.05), f"gain not applied: ratio {ratio:.4f}"


def test_gpt2_mode_default_std_matches_legacy_formula():
    """Unset initializer_range must give N(0, 0.64/d_model) — std 0.8/sqrt(d_model).

    The legacy scheme, not GPT-2's paper 0.02: toy-model training dynamics
    (the grokking demo memorizes vs. stalls) depend on this scale.
    """
    import math

    cfg = _cfg(d_model=128, d_head=32, d_mlp=512, d_vocab=114)
    bridge = TransformerBridge.boot_native(cfg)
    std = bridge.original_model.layers[0].attn.q.weight.std().item()
    assert std == pytest.approx(0.8 / math.sqrt(128), rel=0.05)
