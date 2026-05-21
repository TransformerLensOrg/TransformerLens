"""Bridge vs HookedTransformer parity test for cross-run hook_mlp_in patching.

Issue #1317 fix: bridge fires hook_mlp_in on the pre-ln2 residual (via
BlockBridge's ln2 forward_pre_hook), matching legacy HookedTransformer
semantics. Patches flow through ln2 → mlp on both paths.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

_MODEL = "EleutherAI/pythia-14m"


@pytest.mark.slow
@pytest.mark.parametrize("layer", [0, 3])
def test_cross_run_mlp_in_patch_matches_legacy(layer: int) -> None:
    """Splice cached resid_mid from run A into run B's hook_mlp_in; logits should match."""
    bridge = TransformerBridge.boot_transformers(_MODEL, device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    ht = HookedTransformer.from_pretrained_no_processing(_MODEL, device="cpu")
    ht.cfg.use_hook_mlp_in = True

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

    bridge.run_with_hooks(
        prompt_a, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _cap(cache_a_bridge))]
    )
    ht.run_with_hooks(prompt_a, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _cap(cache_a_ht))])

    assert cache_a_bridge["v"].shape == cache_a_ht["v"].shape
    captured_diff = (cache_a_bridge["v"] - cache_a_ht["v"]).abs().max().item()
    assert (
        captured_diff < 1e-2
    ), f"Bridge hook_mlp_in captures different values than HT: {captured_diff:.3e}"

    bridge_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _patch(cache_a_bridge))]
    )
    ht_logits = ht.run_with_hooks(
        prompt_b, fwd_hooks=[(f"blocks.{layer}.hook_mlp_in", _patch(cache_a_ht))]
    )

    with torch.no_grad():
        baseline_diff = (bridge(prompt_b) - ht(prompt_b)).abs().max().item()
    patched_diff = (bridge_logits - ht_logits).abs().max().item()
    assert patched_diff < 10 * max(baseline_diff, 1e-5), (
        f"Bridge vs HT cross-run mlp_in patch logits diverge {patched_diff:.3e}, "
        f">10x the unhooked baseline {baseline_diff:.3e}"
    )
