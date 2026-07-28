"""Tests for ``build_bridge_from_module`` free function.

Signature, defaults, and behavior must stay aligned with dev-4.x so that the
v4 merge is a no-op for users importing from this module.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.model_bridge.sources.native import (
    NativeModel,
    initialize_native_model,
)


def _build_cfg(**overrides) -> TransformerBridgeConfig:
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
        architecture="TransformerLensNative",
        seed=0,
    )
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def _native(cfg: TransformerBridgeConfig) -> NativeModel:
    m = NativeModel(cfg)
    initialize_native_model(m, cfg)
    return m


def test_returns_bridge_around_native_model():
    cfg = _build_cfg()
    model = _native(cfg)
    bridge = build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)
    assert isinstance(bridge, TransformerBridge)

    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    logits = bridge(inputs, return_type="logits")
    assert logits.shape == (2, cfg.n_ctx, cfg.d_vocab)


def test_requires_exactly_one_config():
    cfg = _build_cfg()
    model = _native(cfg)
    with pytest.raises(ValueError, match="exactly one of hf_config or tl_config"):
        build_bridge_from_module(model, architecture="TransformerLensNative")
    with pytest.raises(ValueError, match="exactly one"):
        build_bridge_from_module(
            model,
            architecture="TransformerLensNative",
            tl_config=cfg,
            hf_config=object(),
        )


def test_dtype_defaults_to_model_first_param():
    cfg = _build_cfg()
    model = _native(cfg).to(dtype=torch.bfloat16)
    bridge = build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)
    assert bridge.cfg.dtype is torch.bfloat16


def test_device_defaults_to_model_first_param():
    cfg = _build_cfg()
    model = _native(cfg)  # CPU by default
    bridge = build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)
    assert "cpu" in bridge.cfg.device.lower()


def test_does_not_mutate_supplied_model_dtype():
    cfg = _build_cfg()
    model = _native(cfg).to(dtype=torch.bfloat16)
    before_dtypes = {p.dtype for p in model.parameters()}
    build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)
    after_dtypes = {p.dtype for p in model.parameters()}
    # The bridge wraps submodules with GeneralizedComponents that re-expose
    # the same parameters under different names; what matters is that no
    # parameter got silently cast to a different dtype.
    assert before_dtypes == after_dtypes == {torch.bfloat16}


def test_post_adapter_hook_runs_before_prepare_model():
    cfg = _build_cfg()
    model = _native(cfg)
    hook_calls: list[str] = []

    def hook(adapter):
        # Adapter must already be selected and have the right type when hook runs.
        hook_calls.append(type(adapter).__name__)

    build_bridge_from_module(
        model,
        architecture="TransformerLensNative",
        tl_config=cfg,
        post_adapter_hook=hook,
    )
    assert hook_calls == ["NativeArchitectureAdapter"]


def test_does_not_mutate_supplied_tl_config():
    """The builder must never mutate the caller's tl_config. Callers commonly
    reuse the same config to build multiple bridges (e.g., different seeds);
    leaking architecture/model_name/dtype/device between calls is a silent
    correctness bug."""
    cfg = _build_cfg()
    # Snapshot every public field we know the builder might touch.
    snapshot = {k: getattr(cfg, k) for k in ("architecture", "model_name", "dtype", "device")}

    model = _native(cfg)
    build_bridge_from_module(
        model,
        architecture="TransformerLensNative",
        tl_config=cfg,
        model_name="caller-named",
        dtype=torch.float16,
    )

    for field, before in snapshot.items():
        after = getattr(cfg, field)
        assert (
            before == after
        ), f"build_bridge_from_module mutated tl_config.{field}: {before!r} -> {after!r}"


def test_two_bridges_from_same_config_are_independent():
    """A training loop pattern: build two bridges from one config with different
    seeds. Bridge B's adapter settings must not leak back through the shared
    config into bridge A's cfg."""
    cfg = _build_cfg()
    model_a = _native(cfg)
    model_b = _native(cfg)

    bridge_a = build_bridge_from_module(
        model_a, architecture="TransformerLensNative", tl_config=cfg, model_name="A"
    )
    bridge_b = build_bridge_from_module(
        model_b, architecture="TransformerLensNative", tl_config=cfg, model_name="B"
    )
    assert bridge_a.cfg.model_name == "A"
    assert bridge_b.cfg.model_name == "B"
    # The two bridges must hold distinct config objects.
    assert bridge_a.cfg is not bridge_b.cfg


def test_run_with_cache_fires_attention_pattern_hook():
    cfg = _build_cfg()
    model = _native(cfg)
    bridge = build_bridge_from_module(model, architecture="TransformerLensNative", tl_config=cfg)
    inputs = torch.randint(0, cfg.d_vocab, (1, cfg.n_ctx))
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    key = "blocks.0.attn.hook_pattern"
    assert key in cache
    assert cache[key].shape == (1, cfg.n_heads, cfg.n_ctx, cfg.n_ctx)


def test_hidden_activation_translates_to_act_fn():
    """transformers 5.x Gemma configs expose only hidden_activation; the
    translation must not fall through to the relu default (compat-mode MLP
    reconstruction reads cfg.act_fn)."""
    from transformers import Gemma3TextConfig

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_config_from_hf,
    )

    hf_cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    cfg = build_bridge_config_from_hf(hf_cfg, "Gemma3ForCausalLM", "tiny", torch.float32)
    assert cfg.act_fn == "gelu_pytorch_tanh"
