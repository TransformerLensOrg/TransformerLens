"""Bridge vs HookedTransformer parity test for cross-run Q/K/V patching.

Deliberate strict-xfail: the bridge forks Q/K/V inputs post-ln1; legacy TL
forks pre-ln1. Pure ablations (zero, mean) are unaffected by the placement,
but a cross-run patch — copy a cached residual from run A into run B's
`hook_q_input` — lands in Q's projection already normed for run A's
distribution on the bridge, and pre-norm-then-re-normed on legacy. The logits
diverge.

This test makes the divergence a load-bearing CI signal. When someone ships
pre-ln1 placement (see docs/rfcs/FOLLOWUP-pre-ln-split-qkv.md), the strict
xfail forces them to flip this test to passing in the same PR.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

_MODEL = "EleutherAI/pythia-14m"


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bridge forks Q/K/V inputs post-ln1; legacy HookedTransformer forks "
        "pre-ln1. Cross-run residual patches land in different coordinate "
        "systems, so logits diverge. Tracked in "
        "docs/rfcs/FOLLOWUP-pre-ln-split-qkv.md — flip to passing in the same "
        "PR that ships pre-ln1 placement."
    ),
)
def test_cross_run_q_input_patch_matches_legacy() -> None:
    """Copy a cached residual from prompt A into hook_q_input on prompt B; bridge and HT logits should match."""
    bridge = TransformerBridge.boot_transformers(_MODEL, device="cpu")
    bridge.enable_compatibility_mode(no_processing=True)
    ht = HookedTransformer.from_pretrained_no_processing(_MODEL, device="cpu")
    bridge.set_use_split_qkv_input(True)
    ht.set_use_split_qkv_input(True)

    prompt_a = torch.arange(1, 9).unsqueeze(0)
    prompt_b = torch.arange(10, 18).unsqueeze(0)

    # Cache a residual from run A (pre-ln on HT; the bridge has no pre-ln hook,
    # so we cache the same conceptual slot — hook_q_input post-ln) and splice
    # it into run B's hook_q_input at layer 0.
    cache_a_bridge: dict = {}
    cache_a_ht: dict = {}

    def cap_bridge(tensor, hook):
        cache_a_bridge["q_in"] = tensor.detach().clone()
        return tensor

    def cap_ht(tensor, hook):
        cache_a_ht["q_in"] = tensor.detach().clone()
        return tensor

    bridge.run_with_hooks(prompt_a, fwd_hooks=[("blocks.0.attn.hook_q_input", cap_bridge)])
    ht.run_with_hooks(prompt_a, fwd_hooks=[("blocks.0.hook_q_input", cap_ht)])

    def patch_bridge(tensor, hook):
        return cache_a_bridge["q_in"]

    def patch_ht(tensor, hook):
        return cache_a_ht["q_in"]

    bridge_logits = bridge.run_with_hooks(
        prompt_b, fwd_hooks=[("blocks.0.attn.hook_q_input", patch_bridge)]
    )
    ht_logits = ht.run_with_hooks(prompt_b, fwd_hooks=[("blocks.0.hook_q_input", patch_ht)])

    assert torch.allclose(bridge_logits, ht_logits, atol=1e-4), (
        f"Bridge vs HT cross-run patch logits diverge: max "
        f"{(bridge_logits - ht_logits).abs().max().item():.3e}"
    )
