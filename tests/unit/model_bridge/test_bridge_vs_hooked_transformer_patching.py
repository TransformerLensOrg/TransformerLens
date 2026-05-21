"""Bridge vs HookedTransformer parity tests for cross-run Q/K/V/attn_in patching.

Issue #1317 fix: bridge forks Q/K/V inputs pre-ln1 via BlockBridge's ln1
forward_pre_hook capture, matching legacy HookedTransformer semantics.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

_MODEL = "EleutherAI/pythia-14m"


def _build_pair() -> tuple[TransformerBridge, HookedTransformer]:
    bridge = TransformerBridge.boot_transformers(_MODEL, device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    ht = HookedTransformer.from_pretrained_no_processing(_MODEL, device="cpu")
    return bridge, ht


def _baseline_logit_diff(
    bridge: TransformerBridge, ht: HookedTransformer, prompt: torch.Tensor
) -> float:
    # Bridge vs HT differ slightly without any hooks (different LayerNorm impls).
    with torch.no_grad():
        return (bridge(prompt) - ht(prompt)).abs().max().item()


def _cross_run_patch_parity(
    bridge: TransformerBridge,
    ht: HookedTransformer,
    bridge_hook_path: str,
    ht_hook_path: str,
) -> None:
    """Cache a hook tensor from prompt A, patch into prompt B on both runtimes; logits should match."""
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

    # Captured tensors should agree to within baseline noise — proves the
    # bridge is reading from the correct coordinate frame, not just that
    # patches happen to flow correctly.
    captured_diff = (cache_a_bridge["v"] - cache_a_ht["v"]).abs().max().item()
    assert captured_diff < 1e-2, (
        f"Bridge {bridge_hook_path} captures different values than HT "
        f"{ht_hook_path}: max diff {captured_diff:.3e}"
    )

    bridge_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(bridge_hook_path, _patch(cache_a_bridge))]
    )
    ht_logits = ht.run_with_hooks(prompt_b, fwd_hooks=[(ht_hook_path, _patch(cache_a_ht))])

    baseline = _baseline_logit_diff(bridge, ht, prompt_b)
    patched = (bridge_logits - ht_logits).abs().max().item()
    assert patched < 10 * max(baseline, 1e-5), (
        f"Bridge vs HT cross-run patch logits diverge {patched:.3e}, "
        f">10x the unhooked baseline {baseline:.3e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("hook_slot", ["q_input", "k_input", "v_input"])
@pytest.mark.parametrize("layer", [0, 3])
def test_split_qkv_cross_run_patch_matches_legacy(hook_slot: str, layer: int) -> None:
    """Each of Q, K, V at multiple layers — independent hook surfaces, shared fork code path."""
    bridge, ht = _build_pair()
    bridge.set_use_split_qkv_input(True)
    ht.set_use_split_qkv_input(True)
    _cross_run_patch_parity(
        bridge,
        ht,
        bridge_hook_path=f"blocks.{layer}.attn.hook_{hook_slot}",
        ht_hook_path=f"blocks.{layer}.hook_{hook_slot}",
    )


@pytest.mark.slow
@pytest.mark.parametrize("layer", [0, 3])
def test_attn_in_cross_run_patch_matches_legacy(layer: int) -> None:
    """The shared attn_in fork uses the same captured pre-LN value, separate from split-QKV."""
    bridge, ht = _build_pair()
    bridge.set_use_attn_in(True)
    ht.set_use_attn_in(True)
    _cross_run_patch_parity(
        bridge,
        ht,
        bridge_hook_path=f"blocks.{layer}.attn.hook_attn_in",
        ht_hook_path=f"blocks.{layer}.hook_attn_in",
    )
