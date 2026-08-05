"""Cross-run hook_mlp_in patching semantics on TransformerBridge (#1317).

Formerly anchored on a live HookedTransformer; now anchored on the intrinsic
legacy contracts: ``hook_mlp_in`` captures a copy of the pre-ln2 residual
(``hook_resid_mid``), self-patching is a logits no-op, cross-patching is
effective, and the pre-ln2 closure fires exactly once (the #1317 bug class).

Parameterized over Pythia (native autograd LN) and GPT-2 (manual LN), and over
``no_processing`` so both folded and unfolded compat-mode setups are covered.
"""
from __future__ import annotations

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
        bridge.set_use_hook_mlp_in(True)
        _bridge_cache[key] = bridge
    return _bridge_cache[key]


@pytest.mark.slow
@pytest.mark.parametrize("no_processing", _NO_PROCESSING)
@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("layer", [0, 3])
def test_cross_run_mlp_in_patch_matches_legacy(model: str, layer: int, no_processing: bool) -> None:
    """hook_mlp_in captures the pre-ln2 residual, self-patch is a no-op, cross-patch works."""
    bridge = _build_bridge(model, no_processing)
    hook_path = f"blocks.{layer}.hook_mlp_in"
    # Parallel-block models (Pythia) have no resid_mid: attn and MLP both read resid_pre.
    resid_mid_path = f"blocks.{layer}.hook_resid_mid"
    if resid_mid_path not in bridge.hook_dict:
        resid_mid_path = f"blocks.{layer}.hook_resid_pre"

    prompt_a = torch.arange(1, 9).unsqueeze(0)
    prompt_b = torch.arange(10, 18).unsqueeze(0)

    cache_a: dict = {}
    fire_count = {"n": 0}

    def _cap(cache: dict, key: str, count: bool = False):
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            if count:
                fire_count["n"] += 1
            cache[key] = tensor.detach().clone()
            return tensor

        return _inner

    def _patch(cache: dict, key: str):
        def _inner(tensor: torch.Tensor, hook: object) -> torch.Tensor:
            return cache[key]

        return _inner

    bridge.run_with_hooks(
        prompt_a,
        fwd_hooks=[
            (hook_path, _cap(cache_a, "mlp_in", count=True)),
            (resid_mid_path, _cap(cache_a, "resid_mid")),
        ],
    )

    # Pins down a silent-miss in the ln2 pre-hook (the #1317 bug class).
    assert fire_count["n"] == 1, (
        f"[{model} no_processing={no_processing}] bridge hook_mlp_in fired "
        f"{fire_count['n']} times, expected exactly 1 (pre-ln2 capture closure)"
    )

    assert cache_a["mlp_in"].shape == cache_a["resid_mid"].shape
    captured_diff = (cache_a["mlp_in"] - cache_a["resid_mid"]).abs().max().item()
    assert captured_diff < 1e-5, (
        f"[{model} no_processing={no_processing}] hook_mlp_in captures a value "
        f"{captured_diff:.3e} away from {resid_mid_path}; it must be a copy of the "
        f"pre-ln2 residual stream"
    )

    # Self-patch no-op: run B's own captured value must flow through identically.
    cache_b: dict = {}
    with torch.no_grad():
        baseline_logits = bridge(prompt_b)
    bridge.run_with_hooks(prompt_b, fwd_hooks=[(hook_path, _cap(cache_b, "mlp_in"))])
    selfpatched_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(hook_path, _patch(cache_b, "mlp_in"))]
    )
    self_diff = (selfpatched_logits - baseline_logits).abs().max().item()
    assert self_diff < 1e-5, (
        f"[{model} no_processing={no_processing}] Self-patching hook_mlp_in moved the "
        f"logits by {self_diff:.3e}"
    )

    # Cross-patch effectiveness: run A's value must change run B's logits.
    crosspatched_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(hook_path, _patch(cache_a, "mlp_in"))]
    )
    cross_diff = (crosspatched_logits - baseline_logits).abs().max().item()
    assert cross_diff > 1e-3, (
        f"[{model} no_processing={no_processing}] Cross-patching hook_mlp_in left the "
        f"logits unchanged ({cross_diff:.3e})"
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
