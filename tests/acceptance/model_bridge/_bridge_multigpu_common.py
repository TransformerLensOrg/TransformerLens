"""Shared constants/helpers for the boot_transformers multi-device acceptance suites.

The suites stay in separate files so each gets a fresh pytest process (HF models
release GPU memory better than inference engines, but sequential multi-boot runs
still fragment). Everything they must agree on lives here so the tolerance band
and boot profile can't drift between them.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MULTIGPU_MARKS = [
    pytest.mark.multigpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="needs >= 2 CUDA devices",
    ),
]

# gpt2 (12 layers, fp32 ~0.5 GB): small enough to boot single + split pairs on any
# 2-GPU box, big enough that a balanced split puts real blocks on both devices.
MODEL = "gpt2"
N_LAYERS = 12
PROMPT = "The quick brown fox jumps over"

# Same kernels on identical GPUs either side of the split, so cross-device parity is
# near-exact — the only difference is where accelerate cuts the graph. 1e-4 matches
# the historical bar (test_multi_gpu_bridge.py's original integration test).
ATOL = RTOL = 1e-4


def boot_single() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cuda:0", dtype=torch.float32)


def boot_multi(**device_kwargs) -> TransformerBridge:
    """Boot MODEL fp32 with the caller's placement kwargs (n_devices=, device_map=...)."""
    return TransformerBridge.boot_transformers(MODEL, dtype=torch.float32, **device_kwargs)


def bridge_pair_fixture(**multi_boot_kwargs):
    """Module fixture: single-device reference then the split bridge under test."""

    @pytest.fixture(scope="module")
    def bridges():
        single = boot_single()
        multi = boot_multi(**multi_boot_kwargs)
        return single, multi

    return bridges


def cuda_indices(bridge: TransformerBridge) -> set:
    return {p.device.index for p in bridge.original_model.parameters() if p.device.type == "cuda"}


def assert_logits_match(single, multi, atol: float = ATOL, rtol: float = RTOL) -> None:
    toks = single.to_tokens(PROMPT)
    l1 = single(toks).detach().float().cpu()
    l2 = multi(toks.to(multi.cfg.device)).detach().float().cpu()
    assert l1.shape == l2.shape
    assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))
    assert torch.allclose(
        l1, l2, atol=atol, rtol=rtol
    ), f"max abs diff {(l1 - l2).abs().max().item():.3e}"


def assert_caches_match(cache1, cache2, atol: float = ATOL, rtol: float = RTOL) -> None:
    """Full key-set equality plus per-hook comparison on CPU — a missing key means a
    hook stopped firing under dispatch, a band miss means the split moved values."""
    assert set(cache1.keys()) == set(cache2.keys())
    for name in cache1:
        t1 = cache1[name].detach().float().cpu()
        t2 = cache2[name].detach().float().cpu()
        assert t1.shape == t2.shape, name
        assert torch.allclose(
            t1, t2, atol=atol, rtol=rtol
        ), f"{name}: max abs diff {(t1 - t2).abs().max().item():.3e}"
