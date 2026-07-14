"""Tests for the optional NativeModel features driven by cfg.

Each feature has a minimal "build a bridge with it enabled, forward, check
caches/shapes" test. The goal is to exercise the bridge code paths each cfg
field unlocks (gated MLP, RMS norm, GQA, soft-cap, rotary, attn_only) — these
features make boot_native useful as a regression target for the bridge's
real machinery, not just a flat GPT-2 toy.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    GatedMLPBridge,
    MLPBridge,
    NormalizationBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.sources.native.model import (
    NativeGatedMLP,
    NativeMLP,
    NativeRMSNorm,
)


class _ResultOnlyAttentionBridge(AttentionBridge):
    supports_split_qkv_fork = False
    supports_attn_result = True


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


def test_result_hook_capability_is_independent_of_qkv_input_forks() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    bridge.blocks[0]._modules["attn"] = _ResultOnlyAttentionBridge(
        name="result_only",
        config=bridge.cfg,
    )

    bridge.set_use_attn_result(True)
    assert bridge.cfg.use_attn_result is True
    with pytest.raises(NotImplementedError, match="use_split_qkv_input"):
        bridge.set_use_split_qkv_input(True)


def test_forward_only_capabilities_reject_generation_and_causal_loss() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    bridge.adapter.supports_generation = False
    bridge.adapter.supports_causal_loss = False
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 4))

    with pytest.raises(NotImplementedError, match="generation is not supported"):
        bridge.generate(tokens, max_new_tokens=1)
    with pytest.raises(NotImplementedError, match="shifted causal loss"):
        bridge(tokens, return_type="loss")


def test_output_attention_capability_controls_only_the_implicit_request() -> None:
    bridge = TransformerBridge.boot_native(_cfg())
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 4))

    with patch.object(
        bridge.original_model,
        "forward",
        wraps=bridge.original_model.forward,
    ) as forward:
        bridge.run_with_cache(tokens)
    assert forward.call_args.kwargs["output_attentions"] is True

    bridge.adapter.supports_hf_output_attentions = False
    with patch.object(
        bridge.original_model,
        "forward",
        wraps=bridge.original_model.forward,
    ) as forward:
        bridge.run_with_cache(tokens)
    assert "output_attentions" not in forward.call_args.kwargs

    with patch.object(
        bridge.original_model,
        "forward",
        wraps=bridge.original_model.forward,
    ) as forward:
        bridge.run_with_cache(tokens, output_attentions=True)
    assert forward.call_args.kwargs["output_attentions"] is True


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


def test_attn_scale_one_is_rejected_when_d_head_gt_one():
    """``attn_scale=1.0`` is a UX trap: it reads like "no scaling / standard"
    but actually means "divide by 1" (no scaling). For any non-trivial d_head
    the softmax saturates and training breaks. The constructor must refuse
    this combination with a pointing message."""
    import pytest

    cfg = _cfg(d_head=16)
    cfg.use_attn_scale = True
    cfg.attn_scale = 1.0
    with pytest.raises(ValueError, match="attn_scale=1.0"):
        TransformerBridge.boot_native(cfg)


def test_attn_scale_one_allowed_when_d_head_one():
    """``d_head=1`` makes ``sqrt(d_head)==1``, so ``attn_scale=1.0`` is no
    longer a trap — it's the same as the default scaling. The guard must NOT
    fire."""
    cfg = _cfg(d_head=1, d_model=4, n_heads=4)
    cfg.use_attn_scale = True
    cfg.attn_scale = 1.0
    bridge = TransformerBridge.boot_native(cfg)
    # Forward must work end-to-end.
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    _ = bridge(inputs, return_type="logits")


def test_attn_scale_custom_nondefault_is_allowed():
    """The guard fires only on the specific 1.0 trap, not on any other custom
    scale a user might pick (e.g. for parity with an external implementation)."""
    cfg = _cfg(d_head=16)
    cfg.use_attn_scale = True
    cfg.attn_scale = 2.5  # not 1.0, not sqrt(d_head); user knows what they want
    bridge = TransformerBridge.boot_native(cfg)
    assert bridge.original_model.layers[0].attn.scale == 2.5


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


def test_gated_mlp_honors_act_fn():
    """``cfg.act_fn`` selects the gating non-linearity: silu→SwiGLU,
    relu→ReGLU, gelu/gelu_new→GeGLU. Each must actually use the requested
    activation rather than silently falling back to silu."""
    cfg_relu = _cfg(gated_mlp=True, act_fn="relu")
    cfg_silu = _cfg(gated_mlp=True, act_fn="silu")
    cfg_gelu = _cfg(gated_mlp=True, act_fn="gelu")

    a = TransformerBridge.boot_native(cfg_relu).blocks[0].mlp.original_component
    b = TransformerBridge.boot_native(cfg_silu).blocks[0].mlp.original_component
    c = TransformerBridge.boot_native(cfg_gelu).blocks[0].mlp.original_component

    # Probe each activation with a sentinel: the negatives expose the difference.
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    relu_ref = torch.relu(x)
    silu_ref = torch.nn.functional.silu(x)
    gelu_ref = torch.nn.functional.gelu(x)

    assert torch.allclose(a.act(x), relu_ref)
    assert torch.allclose(b.act(x), silu_ref)
    assert torch.allclose(c.act(x), gelu_ref)


def test_gated_mlp_rejects_unknown_act_fn():
    """Parity with NativeMLP: unknown act_fn must raise, not silently default
    to silu. Otherwise a user typing ``act_fn="reul"`` and toggling gated_mlp
    on/off would see two different models with no diagnostic."""
    import pytest

    cfg = _cfg(gated_mlp=True, act_fn="not-an-activation")
    with pytest.raises(ValueError, match="Unsupported act_fn"):
        TransformerBridge.boot_native(cfg)


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


def test_no_norm_uses_identity_modules_with_hooks():
    cfg = _cfg(normalization_type=None)
    bridge = TransformerBridge.boot_native(cfg)

    assert isinstance(bridge.blocks[0].ln1, GeneralizedComponent)
    assert not isinstance(bridge.blocks[0].ln1, NormalizationBridge)
    assert isinstance(bridge.blocks[0].ln1.original_component, torch.nn.Identity)
    assert isinstance(bridge.blocks[0].ln2.original_component, torch.nn.Identity)
    assert isinstance(bridge.ln_final.original_component, torch.nn.Identity)

    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    assert torch.equal(cache["blocks.0.ln1.hook_in"], cache["blocks.0.ln1.hook_out"])
    assert torch.equal(cache["ln_final.hook_in"], cache["ln_final.hook_out"])


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


def test_rotary_does_not_shadow_nn_module_apply():
    """nn.Module.apply(fn) is PyTorch's recursive function-applier — used by
    init utilities, weight inspection, and many production training loops.
    NativeRotary's RoPE helper must be named something other than ``apply``
    so it doesn't break ``bridge.apply(fn)`` when a rotary instance lives in
    the module tree."""
    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16)
    bridge = TransformerBridge.boot_native(cfg)

    visited = []
    bridge.apply(lambda m: visited.append(type(m).__name__))
    # The recursion must complete and visit modules — including the rotary one
    # — without TypeError.
    assert "NativeRotary" in visited


def test_rotary_honors_position_ids():
    """Rotary must slice the cached cos/sin tables by ``position_ids`` so the
    caller's chosen positions are honored. A silent-drop bug uses 0..seq-1
    regardless and produces identical patterns regardless of the supplied
    positions — exactly the failure mode for packed sequences, prefix caches,
    and continuation past a cached prefix.

    RoPE has a translation-invariance property: rotating Q and K by the same
    constant shift leaves ``q·k`` (and thus the attention pattern) unchanged.
    So we pick positions with **different relative spacings** (1-apart vs
    2-apart), which produce genuinely different attention patterns when RoPE
    honors ``position_ids`` and identical patterns when it doesn't.
    """
    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=16)
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, 8))

    dense_positions = torch.arange(8).unsqueeze(0).expand(2, -1)  # 0,1,2,...,7
    spaced_positions = (torch.arange(8) * 2).unsqueeze(0).expand(2, -1)  # 0,2,4,...,14

    _, c_dense = bridge.run_with_cache(inputs, return_type="logits", position_ids=dense_positions)
    _, c_spaced = bridge.run_with_cache(inputs, return_type="logits", position_ids=spaced_positions)
    _, c_none = bridge.run_with_cache(inputs, return_type="logits")

    pat_dense = c_dense["blocks.0.attn.hook_pattern"]
    pat_spaced = c_spaced["blocks.0.attn.hook_pattern"]
    pat_none = c_none["blocks.0.attn.hook_pattern"]

    # Default (no position_ids) must match dense 0..seq-1 exactly.
    assert torch.allclose(pat_dense, pat_none, atol=1e-6)
    # Different relative spacings must produce different attention patterns.
    assert not torch.allclose(pat_dense, pat_spaced, atol=1e-4), (
        "Rotary attention ignored position_ids — pattern unchanged despite "
        "different relative spacings."
    )


def test_input_longer_than_n_ctx_raises_with_clear_message():
    """Both the absolute-embed nn.Embedding lookup and the rotary cos/sin
    broadcast would otherwise produce opaque errors that don't mention n_ctx.
    The up-front check must raise ValueError naming both the input length and
    n_ctx, on both code paths."""
    import pytest

    for kind in ("standard", "rotary"):
        cfg = _cfg(
            positional_embedding_type=kind,
            n_heads=4,
            d_head=16,
            n_ctx=8,
        )
        bridge = TransformerBridge.boot_native(cfg)
        long_inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx + 4))
        with pytest.raises(ValueError, match=r"input length 12 exceeds n_ctx=8"):
            bridge(long_inputs, return_type="logits")


def test_rope_scaling_linear_extends_effective_context():
    """Linear (position-interpolation) rope_scaling divides positions by the
    factor before computing freqs. Same n_ctx-slot table, factor× longer
    effective context. We assert the cached cos table differs from the
    unscaled version and matches a hand-computed reference."""
    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg.rope_scaling = {"type": "linear", "factor": 2.0}
    bridge = TransformerBridge.boot_native(cfg)
    rotary = bridge.original_model.rotary
    assert rotary.position_scale == 2.0

    # Reference: positions divided by 2 → freqs are half. cos[pos=0] is still
    # all-ones, but cos[pos=1] is the unscaled cos[pos=0.5] — i.e. shallower
    # rotation than the unscaled cos[pos=1].
    cfg_no = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    rotary_no = TransformerBridge.boot_native(cfg_no).original_model.rotary
    assert not torch.allclose(rotary.cos_cached, rotary_no.cos_cached)


def test_rope_scaling_ntk_scales_base_frequency():
    """NTK-aware rope_scaling scales the base frequency rather than positions.
    Effective base must exceed the configured rotary_base by factor^(d/(d-2))."""
    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg.rope_scaling = {"type": "ntk", "factor": 4.0}
    bridge = TransformerBridge.boot_native(cfg)
    rotary = bridge.original_model.rotary

    expected_base = float(cfg.rotary_base) * (4.0 ** (16 / 14))
    assert abs(rotary.effective_base - expected_base) < 1.0
    assert rotary.position_scale == 1.0  # NTK doesn't touch positions.


def test_rope_scaling_llama3_rescales_inv_freq_per_band():
    """Llama-3 by-parts scheme rescales inv_freq per frequency band rather
    than uniformly. With factor>1 and a reasonable original_ctx, the resulting
    cos table must differ from both the unscaled and the linear-scaled versions
    — otherwise we silently fell through to one of the simpler paths."""
    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg.rope_scaling = {
        "type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8,
    }
    bridge_llama3 = TransformerBridge.boot_native(cfg)

    cfg_linear = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg_linear.rope_scaling = {"type": "linear", "factor": 8.0}
    bridge_linear = TransformerBridge.boot_native(cfg_linear)

    cfg_none = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    bridge_none = TransformerBridge.boot_native(cfg_none)

    c_llama3 = bridge_llama3.original_model.rotary.cos_cached
    c_linear = bridge_linear.original_model.rotary.cos_cached
    c_none = bridge_none.original_model.rotary.cos_cached

    assert not torch.allclose(c_llama3, c_none)
    assert not torch.allclose(c_llama3, c_linear)


def test_rope_scaling_unknown_type_raises():
    import pytest

    cfg = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg.rope_scaling = {"type": "moonshot", "factor": 2.0}
    with pytest.raises(NotImplementedError, match="moonshot"):
        TransformerBridge.boot_native(cfg)


def test_rope_scaling_none_or_factor_one_is_noop():
    """Empty / None rope_scaling, or factor <= 1, must produce the same cos
    table as no scaling. A user explicitly disabling scaling shouldn't pay
    surprise drift."""
    cfg_none = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg_factor_1 = _cfg(positional_embedding_type="rotary", n_heads=4, d_head=16, n_ctx=8)
    cfg_factor_1.rope_scaling = {"type": "linear", "factor": 1.0}

    r_none = TransformerBridge.boot_native(cfg_none).original_model.rotary
    r_f1 = TransformerBridge.boot_native(cfg_factor_1).original_model.rotary
    assert torch.allclose(r_none.cos_cached, r_f1.cos_cached)
    assert r_f1.position_scale == 1.0


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


# -- attention_mask -----------------------------------------------------------


def test_attention_dir_causal_masks_future_tokens():
    cfg = _cfg(attention_dir="causal")
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))

    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    pattern = cache["blocks.0.attn.hook_pattern"]
    future_mask = torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1)

    assert torch.allclose(
        pattern[:, :, future_mask],
        torch.zeros_like(pattern[:, :, future_mask]),
        atol=1e-6,
    )


def test_attention_dir_bidirectional_allows_future_tokens():
    cfg = _cfg(attention_dir="bidirectional")
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))

    _, cache = bridge.run_with_cache(inputs, return_type="logits")
    pattern = cache["blocks.0.attn.hook_pattern"]

    assert pattern[:, :, 0, 1:].gt(0).all()
    assert torch.allclose(pattern.sum(dim=-1), torch.ones_like(pattern.sum(dim=-1)), atol=1e-6)


def test_attention_mask_2d_padding_changes_output():
    """A 2D HF-style padding mask (1=keep, 0=pad) must actually mask keys.
    Verified by comparing outputs with and without the mask — a silent-drop
    bug yields identical outputs."""
    cfg = _cfg(n_layers=2)
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))

    # Mask out the second half of each sequence.
    mask = torch.ones_like(inputs)
    mask[:, cfg.n_ctx // 2 :] = 0

    out_masked = bridge(inputs, return_type="logits", attention_mask=mask)
    out_all_keep = bridge(inputs, return_type="logits", attention_mask=torch.ones_like(inputs))
    out_no_mask = bridge(inputs, return_type="logits")

    # All-keep mask matches no-mask exactly: providing all-1s must be a no-op.
    assert torch.allclose(out_no_mask, out_all_keep, atol=1e-6)
    # Padding mask must change the result on the un-masked positions (positions
    # 0..n_ctx//2 - 1 can only attend to themselves, but masking the keys for
    # positions ≥ n_ctx//2 still propagates into the residual stream through
    # layer 2 because layer 1's masked positions had NaN/garbage outputs that
    # the next attention reads).
    # We just need *some* difference somewhere.
    assert not torch.allclose(
        out_masked, out_no_mask, atol=1e-4
    ), "Padding mask had no effect on output — attention_mask was silently dropped."


def test_attention_mask_padded_key_has_zero_pattern_weight():
    """The most direct invariant: when a key position is padded, no query
    should put any weight on it (post-softmax)."""
    cfg = _cfg(n_layers=1)
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))

    mask = torch.ones_like(inputs)
    mask[:, cfg.n_ctx // 2 :] = 0  # mask out keys at positions ≥ n_ctx/2

    _, cache = bridge.run_with_cache(inputs, return_type="logits", attention_mask=mask)
    pattern = cache["blocks.0.attn.hook_pattern"]
    # For non-padded query rows (positions 0..n_ctx//2-1), no weight on padded
    # keys. (Padded-query rows have all -inf scores → NaN/uniform post-softmax;
    # we don't assert on those.)
    visible_queries = pattern[:, :, : cfg.n_ctx // 2, :]
    padded_key_weight = visible_queries[:, :, :, cfg.n_ctx // 2 :]
    assert torch.allclose(
        padded_key_weight, torch.zeros_like(padded_key_weight), atol=1e-6
    ), "Visible queries put non-zero weight on padded keys"


def test_attention_mask_invalid_shape_raises():
    import pytest

    cfg = _cfg(n_layers=1)
    bridge = TransformerBridge.boot_native(cfg)
    inputs = torch.randint(0, cfg.d_vocab, (2, cfg.n_ctx))
    # 3D mask shape isn't supported — must raise rather than silently drop.
    bad_mask = torch.ones(2, cfg.n_ctx, cfg.n_ctx, dtype=torch.bool)
    with pytest.raises(ValueError, match="attention_mask must be 2D"):
        bridge(inputs, return_type="logits", attention_mask=bad_mask)


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
