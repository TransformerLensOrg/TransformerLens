"""Bridge vs HookedTransformer parity test for cross-run hook_mlp_in patching (#1317).

Parameterized over Pythia (native autograd LN) and GPT-2 (manual LN), and over
``no_processing`` so both folded and unfolded compat-mode setups are covered.
"""
from __future__ import annotations

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
        bridge.set_use_hook_mlp_in(True)
        ht.cfg.use_hook_mlp_in = True
        _pair_cache[key] = (bridge, ht)
    return _pair_cache[key]


def _baseline_logit_diff(model: str, no_processing: bool, prompt: torch.Tensor) -> float:
    key = (model, no_processing, tuple(prompt.flatten().tolist()))
    if key not in _baseline_cache:
        bridge, ht = _build_pair(model, no_processing)
        with torch.no_grad():
            _baseline_cache[key] = (bridge(prompt) - ht(prompt)).abs().max().item()
    return _baseline_cache[key]


@pytest.mark.slow
@pytest.mark.parametrize("no_processing", _NO_PROCESSING)
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("layer", [0, 3])
def test_cross_run_mlp_in_patch_matches_legacy(model: str, layer: int, no_processing: bool) -> None:
    """Splice cached resid_mid from run A into run B's hook_mlp_in; logits should match."""
    bridge, ht = _build_pair(model, no_processing)

    prompt_a = torch.arange(1, 9).unsqueeze(0)
    prompt_b = torch.arange(10, 18).unsqueeze(0)

    cache_a_bridge: dict = {}
    cache_a_ht: dict = {}
    bridge_fire_count = {"n": 0}

    def _cap_bridge(tensor: torch.Tensor, hook: object) -> torch.Tensor:
        bridge_fire_count["n"] += 1
        cache_a_bridge["v"] = tensor.detach().clone()
        return tensor

    def _cap_ht(tensor: torch.Tensor, hook: object) -> torch.Tensor:
        cache_a_ht["v"] = tensor.detach().clone()
        return tensor

    def _patch(cache: dict) -> "object":
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            return cache["v"]

        return _inner

    bridge.run_with_hooks(prompt_a, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _cap_bridge)])
    ht.run_with_hooks(prompt_a, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _cap_ht)])

    # Pins down a silent-miss in the ln2 pre-hook (the #1317 bug class).
    assert bridge_fire_count["n"] == 1, (
        f"[{model} no_processing={no_processing}] bridge hook_mlp_in fired "
        f"{bridge_fire_count['n']} times, expected exactly 1 (pre-ln2 capture closure)"
    )

    assert cache_a_bridge["v"].shape == cache_a_ht["v"].shape
    captured_diff = (cache_a_bridge["v"] - cache_a_ht["v"]).abs().max().item()
    assert captured_diff < 1e-3, (
        f"[{model} no_processing={no_processing}] Bridge hook_mlp_in captures "
        f"different values than HT: {captured_diff:.3e}"
    )

    bridge_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _patch(cache_a_bridge))]
    )
    ht_logits = ht.run_with_hooks(
        prompt_b, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _patch(cache_a_ht))]
    )

    baseline_diff = _baseline_logit_diff(model, no_processing, prompt_b)
    patched_diff = (bridge_logits - ht_logits).abs().max().item()
    assert patched_diff < 10 * max(baseline_diff, 1e-5), (
        f"[{model} no_processing={no_processing}] Bridge vs HT cross-run mlp_in patch "
        f"logits diverge {patched_diff:.3e}, >10x the unhooked baseline {baseline_diff:.3e}"
    )


@pytest.mark.slow
def test_mlp_in_gated_off_does_not_fire() -> None:
    """When ``use_hook_mlp_in`` is False, the bridge pre-ln2 closure must skip firing."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    bridge.set_use_hook_mlp_in(False)

    fire_count = {"n": 0}

    def _counter(tensor: torch.Tensor, hook: object) -> torch.Tensor:
        fire_count["n"] += 1
        return tensor

    prompt = torch.arange(1, 9).unsqueeze(0)
    bridge.run_with_hooks(prompt, fwd_hooks=[("blocks.0.hook_mlp_in", _counter)])
    assert fire_count["n"] == 0, (
        f"hook_mlp_in fired {fire_count['n']} times with use_hook_mlp_in=False; "
        "should not fire when the flag is off"
    )
