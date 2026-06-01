"""Tests for the optional NativeModel features driven by cfg.

Each feature has a minimal "build a bridge with it enabled, forward, check
caches/shapes" test. The goal is to exercise the bridge code paths each cfg
field unlocks (gated MLP, RMS norm, GQA, soft-cap, rotary, attn_only) — these
features make boot_native useful as a regression target for the bridge's
real machinery, not just a flat GPT-2 toy.
"""
from __future__ import annotations

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    MLPBridge,
    NormalizationBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.sources.native.model import (
    NativeGatedMLP,
    NativeMLP,
    NativeRMSNorm,
)


def _cfg(**overrides) -> TransformerBridgeConfig:
    base = dict(
        d_model=32,
        d_head=16,
        n_heads=4,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="silu",
        normalization_type="LN",
        seed=0,
    )
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def _forward(bridge: TransformerBridge) -> torch.Tensor:
    inputs = torch.randint(0, bridge.cfg.d_vocab, (2, bridge.cfg.n_ctx))
    return bridge(inputs, return_type="logits")


# -- soft-cap -----------------------------------------------------------------


def test_attn_scores_soft_cap_bounds_pattern():
    """When attn_scores_soft_cap is set, no entry of the soft-cap target should
    blow up. We assert the pattern (post-softmax) is still well-formed: rows
    sum to 1 and no nan/inf. The cap itself happens pre-softmax, but a buggy
    application (e.g. wrong sign) shows up immediately downstream."""
    cfg = _cfg(attn_scores_soft_cap=30.0)
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    pattern = cache["blocks.0.attn.hook_pattern"]
    assert torch.allclose(pattern.sum(dim=-1), torch.ones_like(pattern.sum(dim=-1)), atol=1e-5)
    assert torch.isfinite(pattern).all()


def test_output_logits_soft_cap_bounds_logits():
    """Logits must be bounded by ±cap when output_logits_soft_cap > 0."""
    cap = 5.0
    cfg = _cfg(output_logits_soft_cap=cap, seed=0)
    # Force-large outputs by skipping the soft-cap → then re-enabling. Easier:
    # just pick a cap and assert |logits| <= cap. tanh-cap math guarantees it.
    bridge = TransformerBridge.boot_native(cfg)
    logits = _forward(bridge)
    assert logits.abs().max().item() <= cap + 1e-5


# -- gated MLP ----------------------------------------------------------------


def test_gated_mlp_swaps_module_class_and_bridge():
    """With gated_mlp=True, the underlying MLP must be NativeGatedMLP and the
    adapter must wrap it in GatedMLPBridge (not the plain MLPBridge).

    Bridge setup replaces original_model submodules with bridge wrappers, so
    we peek at the underlying module via ``original_component``.
    """
    cfg = _cfg(gated_mlp=True, act_fn="silu")
    bridge = TransformerBridge.boot_native(cfg)
    mlp_bridge = bridge.blocks[0].mlp
    assert isinstance(mlp_bridge, GatedMLPBridge)
    assert isinstance(mlp_bridge.original_component, NativeGatedMLP)
    _ = _forward(bridge)  # forward must work end-to-end


def test_gated_mlp_default_is_plain():
    cfg = _cfg()
    bridge = TransformerBridge.boot_native(cfg)
    mlp_bridge = bridge.blocks[0].mlp
    # GatedMLPBridge subclasses MLPBridge, so the negative check uses the
    # subclass type directly.
    assert isinstance(mlp_bridge, MLPBridge)
    assert not isinstance(mlp_bridge, GatedMLPBridge)
    assert isinstance(mlp_bridge.original_component, NativeMLP)
    assert not isinstance(mlp_bridge.original_component, NativeGatedMLP)


# -- RMS norm -----------------------------------------------------------------


def test_rms_norm_computes_variance_in_fp32():
    """HF Llama's RMSNorm upcasts to fp32 to compute variance, then casts back.
    NativeRMSNorm must match that pattern so bf16 / fp16 parity comparisons
    against a Llama reference don't drift.

    We compare bf16-cast norm output against the fp32 reference on the same
    seeded input. With the fp32 variance pre-step, the bf16 result lands close
    to the reference; a naive bf16-only variance would blow this bound.
    """
    torch.manual_seed(0)
    norm_fp32 = NativeRMSNorm(d_model=64, eps=1e-5)
    # Cast a copy of the norm to bf16 — this is the "model cast to bf16"
    # scenario where weight is bf16 and inputs are bf16. Output then follows
    # the input dtype, matching HF Llama.
    norm_bf16 = NativeRMSNorm(d_model=64, eps=1e-5).to(torch.bfloat16)
    norm_bf16.weight.data.copy_(norm_fp32.weight.to(torch.bfloat16))

    x = torch.randn(2, 8, 64)
    out_fp32 = norm_fp32(x)
    out_bf16 = norm_bf16(x.to(torch.bfloat16))

    assert out_fp32.dtype is torch.float32
    assert out_bf16.dtype is torch.bfloat16
    # bf16 has ~8 bits of mantissa; the gap from a fp32-computed reference is
    # in the 1e-2 range. A naive bf16-only RMS would drift well beyond this.
    drift = (out_bf16.float() - out_fp32).abs().max().item()
    assert drift < 5e-2, f"RMSNorm bf16 drifted {drift!r} from fp32 reference"


def test_rms_norm_swaps_module_class_and_bridge():
    cfg = _cfg(normalization_type="RMS")
    bridge = TransformerBridge.boot_native(cfg)
    ln1_bridge = bridge.blocks[0].ln1
    ln_final_bridge = bridge.ln_final
    assert isinstance(ln1_bridge, RMSNormalizationBridge)
    assert isinstance(ln1_bridge.original_component, NativeRMSNorm)
    assert isinstance(ln_final_bridge, RMSNormalizationBridge)
    assert isinstance(ln_final_bridge.original_component, NativeRMSNorm)
    _ = _forward(bridge)


def test_final_rms_only_swaps_the_final_norm():
    """final_rms=True with normalization_type='LN' uses LN in blocks but RMS
    for the final norm. Matches the Llama config semantic."""
    cfg = _cfg(normalization_type="LN", final_rms=True)
    bridge = TransformerBridge.boot_native(cfg)
    ln1_bridge = bridge.blocks[0].ln1
    ln_final_bridge = bridge.ln_final
    # Blocks use plain LN.
    assert isinstance(ln1_bridge, NormalizationBridge)
    assert not isinstance(ln1_bridge, RMSNormalizationBridge)
    assert isinstance(ln1_bridge.original_component, torch.nn.LayerNorm)
    # Final norm is RMS.
    assert isinstance(ln_final_bridge, RMSNormalizationBridge)
    assert isinstance(ln_final_bridge.original_component, NativeRMSNorm)
    _ = _forward(bridge)


def test_ln_default_uses_layernorm():
    cfg = _cfg()
    bridge = TransformerBridge.boot_native(cfg)
    ln1_bridge = bridge.blocks[0].ln1
    assert isinstance(ln1_bridge, NormalizationBridge)
    assert not isinstance(ln1_bridge, RMSNormalizationBridge)
    assert isinstance(ln1_bridge.original_component, torch.nn.LayerNorm)


# -- GQA ----------------------------------------------------------------------


def test_gqa_shapes_kv_smaller_than_q():
    """With n_key_value_heads < n_heads, K/V projections must produce fewer
    heads than Q, and the model must still produce the right-shaped logits.

    cache shapes follow Q's head count after repeat-expansion, so hook_pattern
    is [batch, n_heads, seq, seq] regardless of n_kv_heads."""
    cfg = _cfg(n_heads=4, n_key_value_heads=2)
    bridge = TransformerBridge.boot_native(cfg)
    attn = bridge.original_model.layers[0].attn
    # Q gets full head dim, K/V get half.
    assert attn.q.out_features == cfg.n_heads * cfg.d_head
    assert attn.k.out_features == cfg.n_key_value_heads * cfg.d_head
    assert attn.v.out_features == cfg.n_key_value_heads * cfg.d_head

    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    pattern = cache["blocks.0.attn.hook_pattern"]
    assert pattern.shape == (2, cfg.n_heads, cfg.n_ctx, cfg.n_ctx)


def test_gqa_default_is_full_mha():
    """Without n_key_value_heads (default None), K/V have the same head count
    as Q."""
    cfg = _cfg(n_heads=4)
    bridge = TransformerBridge.boot_native(cfg)
    attn = bridge.original_model.layers[0].attn
    assert attn.n_kv_heads == cfg.n_heads
    assert attn.k.out_features == cfg.n_heads * cfg.d_head


# -- attn_only ----------------------------------------------------------------


def test_attn_only_skips_mlp_branch():
    """attn_only=True must drop the MLP/ln2 entirely. The bridge mapping
    omits the mlp slot, cache contains no mlp-related hooks, and bridge
    construction emits no ``hook_resid_mid``/``hook_mlp_out`` alias warnings
    (those are dropped from BlockBridge.hook_aliases under attn_only)."""
    import warnings

    cfg = _cfg(attn_only=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge = TransformerBridge.boot_native(cfg)
    stale_alias_warnings = [
        w for w in caught if "hook_resid_mid" in str(w.message) or "hook_mlp_out" in str(w.message)
    ]
    assert stale_alias_warnings == [], (
        "attn_only must drop the ln2/mlp-targeting hook aliases so "
        f"_register_aliases stays quiet, got: {[str(w.message) for w in stale_alias_warnings]}"
    )

    block = bridge.original_model.layers[0]
    assert not hasattr(block, "mlp")
    assert not hasattr(block, "ln2")
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    mlp_keys = [k for k in cache.keys() if ".mlp." in k or k.endswith(".mlp")]
    assert mlp_keys == [], f"attn_only should fire no MLP hooks, got {mlp_keys}"


# -- rotary -------------------------------------------------------------------


def test_rotary_drops_pos_embed_and_forward_works():
    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16)
    bridge = TransformerBridge.boot_native(cfg)
    # The native model must not allocate a learned position embedding when
    # rotary is in effect.
    assert bridge.original_model.pos is None
    assert bridge.original_model.rotary is not None
    # And the bridge's pos_embed component slot is absent.
    assert "pos_embed" not in bridge.adapter.component_mapping
    _ = _forward(bridge)


def test_rotary_pattern_differs_from_absolute():
    """Sanity that rotary actually changes the attention pattern relative to
    absolute embeddings — would catch silent no-op (e.g. cos/sin buffers
    wired wrong)."""
    base = dict(
        d_model=32,
        d_head=16,
        n_heads=4,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="silu",
        normalization_type="LN",
        seed=0,
    )
    bridge_abs = TransformerBridge.boot_native(TransformerBridgeConfig(**base))
    bridge_rope = TransformerBridge.boot_native(
        TransformerBridgeConfig(**{**base, "positional_embedding_type": "rotary"})
    )
    inputs = torch.randint(0, base["d_vocab"], (2, base["n_ctx"]))
    _, c_abs = bridge_abs.run_with_cache(inputs, return_type="logits")
    _, c_rope = bridge_rope.run_with_cache(inputs, return_type="logits")
    assert not torch.allclose(
        c_abs["blocks.0.attn.hook_pattern"], c_rope["blocks.0.attn.hook_pattern"]
    )


# -- init modes ---------------------------------------------------------------


import pytest


@pytest.mark.parametrize(
    "init_mode",
    ["gpt2", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
)
def test_init_mode_builds_and_forwards(init_mode):
    """Each supported init mode must build a working bridge and run forward
    without numerical disasters (no NaN/Inf in logits)."""
    cfg = _cfg(init_mode=init_mode, seed=0)
    bridge = TransformerBridge.boot_native(cfg)
    logits = _forward(bridge)
    assert torch.isfinite(logits).all(), f"init_mode={init_mode} produced non-finite logits"


@pytest.mark.parametrize(
    "init_mode",
    ["gpt2", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
)
def test_init_mode_seed_reproducible(init_mode):
    """Same seed + same mode → identical parameters."""
    cfg_a = _cfg(init_mode=init_mode, seed=7)
    cfg_b = _cfg(init_mode=init_mode, seed=7)
    a = TransformerBridge.boot_native(cfg_a)
    b = TransformerBridge.boot_native(cfg_b)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.allclose(pa, pb), f"init_mode={init_mode} not reproducible at {na}"


def test_init_mode_rejects_unknown():
    """Unsupported modes fail with a clear list of what IS supported."""
    cfg = _cfg(init_mode="he_kaiser_alpha_quadratic")
    with pytest.raises(NotImplementedError, match="Supported modes"):
        TransformerBridge.boot_native(cfg)


def test_init_modes_diverge_from_each_other():
    """Different init modes must produce visibly different parameter tensors
    under the same seed — otherwise the dispatch is broken."""
    seed = 11
    gpt2 = TransformerBridge.boot_native(_cfg(init_mode="gpt2", seed=seed))
    xav = TransformerBridge.boot_native(_cfg(init_mode="xavier_normal", seed=seed))
    kai = TransformerBridge.boot_native(_cfg(init_mode="kaiming_normal", seed=seed))
    # The bridge stores original_model in __dict__ (not as a child module),
    # so the embedding parameter shows up under tok_embed._original_component.
    embed_key = "tok_embed._original_component.weight"
    g_w = dict(gpt2.named_parameters())[embed_key]
    x_w = dict(xav.named_parameters())[embed_key]
    k_w = dict(kai.named_parameters())[embed_key]
    assert not torch.allclose(g_w, x_w)
    assert not torch.allclose(g_w, k_w)
    assert not torch.allclose(x_w, k_w)


# -- combo --------------------------------------------------------------------


def test_llama_shaped_config_works_end_to_end():
    """The interesting combination — RMS norm + rotary + gated MLP + GQA + no
    learned pos embed + final_rms — is the Llama-3 shape. Exercises all the
    feature switches at once, ensuring they compose."""
    cfg = _cfg(
        n_heads=4,
        n_key_value_heads=2,
        d_head=16,
        normalization_type="RMS",
        final_rms=True,
        gated_mlp=True,
        act_fn="silu",
        positional_embedding_type="rotary",
    )
    bridge = TransformerBridge.boot_native(cfg)
    # Inspect bridge components and their underlying NativeModel modules.
    assert isinstance(bridge.blocks[0].mlp, GatedMLPBridge)
    assert isinstance(bridge.blocks[0].mlp.original_component, NativeGatedMLP)
    assert isinstance(bridge.blocks[0].ln1, RMSNormalizationBridge)
    assert isinstance(bridge.blocks[0].ln1.original_component, NativeRMSNorm)
    assert isinstance(bridge.ln_final, RMSNormalizationBridge)
    assert isinstance(bridge.ln_final.original_component, NativeRMSNorm)
    # Rotary: no learned pos embed on the bridge or under it.
    assert "pos_embed" not in bridge.adapter.component_mapping
    assert bridge.original_model.pos is None
    # GQA: K/V heads independently configured.
    attn = bridge.blocks[0].attn.original_component
    assert attn.n_kv_heads == 2

    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    logits, cache = bridge.run_with_cache(inputs, return_type="logits")
    assert logits.shape == (2, cfg.n_ctx, cfg.d_vocab)
    assert cache["blocks.0.attn.hook_pattern"].shape == (2, cfg.n_heads, cfg.n_ctx, cfg.n_ctx)
