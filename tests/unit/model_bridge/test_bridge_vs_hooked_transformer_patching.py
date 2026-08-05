"""Cross-run Q/K/V/attn_in patching semantics on TransformerBridge (#1317).

Formerly anchored on a live HookedTransformer; now anchored on the intrinsic
legacy contracts, which need no reference model:
- ``hook_{q,k,v}_input`` / ``hook_attn_in`` capture a copy of the pre-ln1
  residual stream (``hook_resid_pre``) — the value HT exposed there.
- Patching a run's own captured value back in is a logits no-op.
- Patching a different run's value in changes the logits.

Parameterized over Pythia (native autograd LN) and GPT-2 (manual LN), and over
``no_processing`` so both folded and unfolded compat-mode setups are covered.
"""
from __future__ import annotations

import os

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

_MODELS = ("EleutherAI/pythia-14m", "gpt2")
_NO_PROCESSING = (True, False)

_bridge_cache: dict[tuple[str, bool], TransformerBridge] = {}


def _build_bridge(model: str, no_processing: bool) -> TransformerBridge:
    key = (model, no_processing)
    if key not in _bridge_cache:
        bridge = TransformerBridge.boot_transformers(model, device="cpu")
        bridge.enable_compatibility_mode(no_processing=no_processing)
        _bridge_cache[key] = bridge
    return _bridge_cache[key]


def _cross_run_patch_semantics(
    bridge: TransformerBridge,
    model: str,
    no_processing: bool,
    hook_path: str,
    capture_tol: float = 1e-5,
) -> None:
    """Verify capture source, self-patch no-op, and cross-patch effectiveness."""
    prompt_a = torch.arange(1, 9).unsqueeze(0)
    prompt_b = torch.arange(10, 18).unsqueeze(0)
    layer = int(hook_path.split(".")[1])
    resid_pre_path = f"blocks.{layer}.hook_resid_pre"

    def _cap(cache: dict, key: str):
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            cache[key] = tensor.detach().clone()
            return tensor

        return _inner

    def _patch(cache: dict, key: str):
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            return cache[key]

        return _inner

    # 1. The gated hook captures a copy of the pre-ln1 residual (HT's contract).
    cache_a: dict = {}
    bridge.run_with_hooks(
        prompt_a,
        fwd_hooks=[(hook_path, _cap(cache_a, "gated")), (resid_pre_path, _cap(cache_a, "resid"))],
    )
    # Split-QKV / attn_in hooks are per-head broadcast copies of the residual:
    # [batch, pos, n_heads, d_model] where every head slice equals resid_pre.
    gated, resid = cache_a["gated"], cache_a["resid"]
    if gated.ndim == resid.ndim + 1:
        resid = resid.unsqueeze(2).expand_as(gated)
    captured_diff = (gated - resid).abs().max().item()
    assert captured_diff < capture_tol, (
        f"[{model} no_processing={no_processing}] {hook_path} captures a value "
        f"{captured_diff:.3e} away from {resid_pre_path}; it must be a per-head copy "
        f"of the pre-ln1 residual stream"
    )

    # 2. Self-patch is a no-op: feeding back run B's own captured value leaves logits intact.
    cache_b: dict = {}
    with torch.no_grad():
        baseline_logits = bridge(prompt_b)
    bridge.run_with_hooks(prompt_b, fwd_hooks=[(hook_path, _cap(cache_b, "gated"))])
    selfpatched_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(hook_path, _patch(cache_b, "gated"))]
    )
    self_diff = (selfpatched_logits - baseline_logits).abs().max().item()
    assert self_diff < 1e-5, (
        f"[{model} no_processing={no_processing}] Self-patching {hook_path} moved the "
        f"logits by {self_diff:.3e}; the patched value must flow through identically"
    )

    # 3. Cross-patch is effective: run A's value changes run B's logits.
    crosspatched_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(hook_path, _patch(cache_a, "gated"))]
    )
    cross_diff = (crosspatched_logits - baseline_logits).abs().max().item()
    assert cross_diff > 1e-3, (
        f"[{model} no_processing={no_processing}] Cross-patching {hook_path} left the "
        f"logits unchanged ({cross_diff:.3e}); the patch is not being applied"
    )


@pytest.mark.slow
@pytest.mark.parametrize("no_processing", _NO_PROCESSING)
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("hook_slot", ["q_input", "k_input", "v_input"])
@pytest.mark.parametrize("layer", [0, 3])
def test_split_qkv_cross_run_patch_matches_legacy(
    model: str, hook_slot: str, layer: int, no_processing: bool
) -> None:
    """Each of Q, K, V at multiple layers, on both native/manual LN paths and folded/unfolded compat."""
    bridge = _build_bridge(model, no_processing)
    bridge.set_use_split_qkv_input(True)
    try:
        _cross_run_patch_semantics(
            bridge, model, no_processing, hook_path=f"blocks.{layer}.attn.hook_{hook_slot}"
        )
    finally:
        bridge.set_use_split_qkv_input(False)


@pytest.mark.slow
@pytest.mark.parametrize("no_processing", _NO_PROCESSING)
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("layer", [0, 3])
def test_attn_in_cross_run_patch_matches_legacy(
    model: str, layer: int, no_processing: bool
) -> None:
    """The shared attn_in fork uses the same captured pre-LN value, separate from split-QKV."""
    bridge = _build_bridge(model, no_processing)
    bridge.set_use_split_qkv_input(False)
    bridge.set_use_attn_in(True)
    try:
        _cross_run_patch_semantics(
            bridge, model, no_processing, hook_path=f"blocks.{layer}.attn.hook_attn_in"
        )
    finally:
        bridge.set_use_attn_in(False)


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_OLMO2_GAP_TEST", "") != "1",
    reason="Set RUN_OLMO2_GAP_TEST=1 to exercise the OLMo-2 post-norm gap (1B-param download).",
)
@pytest.mark.xfail(
    strict=True,
    reason="OLMo 2 post-norm: ln1 maps to post_attention_layernorm, so pre-ln1 capture "
    "reads post-attention residual. Flip to passing when the carve-out is fixed.",
)
def test_olmo2_pre_ln_capture_known_gap() -> None:
    bridge = TransformerBridge.boot_transformers("allenai/OLMo-2-0425-1B", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    bridge.set_use_split_qkv_input(True)
    _cross_run_patch_semantics(
        bridge, "allenai/OLMo-2-0425-1B", True, hook_path="blocks.0.attn.hook_q_input"
    )
