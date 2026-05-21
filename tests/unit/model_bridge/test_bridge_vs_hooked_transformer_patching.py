"""Bridge vs HookedTransformer parity tests for cross-run Q/K/V/attn_in patching (#1317).

Parameterized over Pythia (native autograd LN) and GPT-2 (manual LN), and over
``no_processing`` so both folded and unfolded compat-mode setups are covered.
"""
from __future__ import annotations

import os

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

_MODELS = ("EleutherAI/pythia-14m", "gpt2")
_NO_PROCESSING = (True, False)

_pair_cache: dict[tuple[str, bool], tuple[TransformerBridge, HookedTransformer]] = {}
_baseline_cache: dict[tuple[str, bool, tuple[int, ...]], float] = {}


def _build_pair(model: str, no_processing: bool) -> tuple[TransformerBridge, HookedTransformer]:
    key = (model, no_processing)
    if key not in _pair_cache:
        bridge = TransformerBridge.boot_transformers(model, device="cpu")
        bridge.enable_compatibility_mode(no_processing=no_processing)
        if no_processing:
            ht = HookedTransformer.from_pretrained_no_processing(model, device="cpu")
        else:
            ht = HookedTransformer.from_pretrained(model, device="cpu")
        _pair_cache[key] = (bridge, ht)
    return _pair_cache[key]


def _baseline_logit_diff(model: str, no_processing: bool, prompt: torch.Tensor) -> float:
    key = (model, no_processing, tuple(prompt.flatten().tolist()))
    if key not in _baseline_cache:
        bridge, ht = _build_pair(model, no_processing)
        with torch.no_grad():
            _baseline_cache[key] = (bridge(prompt) - ht(prompt)).abs().max().item()
    return _baseline_cache[key]


def _cross_run_patch_parity(
    model: str,
    no_processing: bool,
    bridge_hook_path: str,
    ht_hook_path: str,
    capture_tol: float = 1e-3,
) -> None:
    """Cache hook tensor from prompt A, patch into prompt B on both runtimes; logits should match."""
    bridge, ht = _build_pair(model, no_processing)
    prompt_a = torch.arange(1, 9).unsqueeze(0)
    prompt_b = torch.arange(10, 18).unsqueeze(0)

    cache_a_bridge: dict = {}
    cache_a_ht: dict = {}

    def _cap(cache: dict) -> "object":
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            cache["v"] = tensor.detach().clone()
            return tensor

        return _inner

    def _patch(cache: dict) -> "object":
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            return cache["v"]

        return _inner

    bridge.run_with_hooks(prompt_a, fwd_hooks=[(bridge_hook_path, _cap(cache_a_bridge))])
    ht.run_with_hooks(prompt_a, fwd_hooks=[(ht_hook_path, _cap(cache_a_ht))])

    captured_diff = (cache_a_bridge["v"] - cache_a_ht["v"]).abs().max().item()
    assert captured_diff < capture_tol, (
        f"[{model} no_processing={no_processing}] Bridge {bridge_hook_path} captures "
        f"different values than HT {ht_hook_path}: max diff {captured_diff:.3e} "
        f"(tol {capture_tol:.0e})"
    )

    bridge_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(bridge_hook_path, _patch(cache_a_bridge))]
    )
    ht_logits = ht.run_with_hooks(prompt_b, fwd_hooks=[(ht_hook_path, _patch(cache_a_ht))])

    baseline = _baseline_logit_diff(model, no_processing, prompt_b)
    patched = (bridge_logits - ht_logits).abs().max().item()
    assert patched < 10 * max(baseline, 1e-5), (
        f"[{model} no_processing={no_processing}] Bridge vs HT cross-run patch logits "
        f"diverge {patched:.3e}, >10x the unhooked baseline {baseline:.3e}"
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
    bridge, ht = _build_pair(model, no_processing)
    bridge.set_use_split_qkv_input(True)
    ht.set_use_split_qkv_input(True)
    _cross_run_patch_parity(
        model,
        no_processing,
        bridge_hook_path=f"blocks.{layer}.attn.hook_{hook_slot}",
        ht_hook_path=f"blocks.{layer}.hook_{hook_slot}",
    )


@pytest.mark.slow
@pytest.mark.parametrize("no_processing", _NO_PROCESSING)
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("layer", [0, 3])
def test_attn_in_cross_run_patch_matches_legacy(
    model: str, layer: int, no_processing: bool
) -> None:
    """The shared attn_in fork uses the same captured pre-LN value, separate from split-QKV."""
    bridge, ht = _build_pair(model, no_processing)
    bridge.set_use_split_qkv_input(False)
    bridge.set_use_attn_in(True)
    ht.set_use_split_qkv_input(False)
    ht.set_use_attn_in(True)
    _cross_run_patch_parity(
        model,
        no_processing,
        bridge_hook_path=f"blocks.{layer}.attn.hook_attn_in",
        ht_hook_path=f"blocks.{layer}.hook_attn_in",
    )


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
    ht = HookedTransformer.from_pretrained_no_processing("allenai/OLMo-2-0425-1B", device="cpu")
    bridge.set_use_split_qkv_input(True)
    ht.set_use_split_qkv_input(True)
    _cross_run_patch_parity(
        "allenai/OLMo-2-0425-1B",
        True,
        bridge_hook_path="blocks.0.attn.hook_q_input",
        ht_hook_path="blocks.0.hook_q_input",
    )
