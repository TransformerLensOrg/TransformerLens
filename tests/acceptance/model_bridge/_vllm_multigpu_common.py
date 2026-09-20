"""Shared constants/helpers for the vLLM multi-GPU acceptance suites.

The suites stay in separate files — each boots its own engine(s) and in-process
vLLM engines under-release GPU memory, so every file gets a fresh pytest process
(see each file's docstring for its run command). Everything they must agree on
lives here so the tolerance band and boot profile can't drift between them.
"""
from __future__ import annotations

import pytest
import torch

MULTIGPU_MARKS = [
    pytest.mark.multigpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="needs >= 2 CUDA devices",
    ),
]

# Qwen2.5-0.5B: 14 attention heads / 2 KV heads — both divisible by TP=2.
# (SmolLM2-135M has 9 heads and cannot tensor-parallelize; keep any
# replacement model even-headed.) 24 layers -> 12 per PP=2 stage.
MODEL = "Qwen/Qwen2.5-0.5B"
N_LAYERS = 24
PROMPT_IDS = [504, 4674, 1442, 29892, 322]  # fixed ids: no tokenizer variance in scope

# vLLM kernel scheduling differs across parallel layouts (all-reduce / send-recv
# order), so exact equality is not expected; band matches scripts/vllm_parity_report.py,
# including its scale-aware atol (final-layer streams reach O(1e3) and noise grows
# with scale).
ATOL = RTOL = 2e-2
REL_BAND = 2e-3


def close(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    """Scale-aware allclose: atol widens with the reference tensor's magnitude."""
    atol_eff = max(ATOL, REL_BAND * t2.abs().max().item())
    return torch.allclose(t1, t2, atol=atol_eff, rtol=RTOL)


def boot_multigpu(**parallel_kwargs):
    """Boot MODEL with the suites' shared fp32 profile; parallel sizes per test."""
    from transformer_lens.model_bridge.sources.vllm.source import boot_vllm

    return boot_vllm(
        MODEL,
        dtype=torch.float32,
        max_model_len=2048,
        gpu_memory_utilization=0.35,
        **parallel_kwargs,
    )


def bridge_pair_fixture(**test_boot_kwargs):
    """Module fixture: single-rank reference then the parallel bridge under test;
    serial boots (never two engines booting at once), closed in reverse."""

    @pytest.fixture(scope="module")
    def bridges():
        b1 = boot_multigpu()
        b2 = boot_multigpu(**test_boot_kwargs)
        yield b1, b2
        b2.close()
        b1.close()

    return bridges


def assert_caches_match(cache1, cache2) -> None:
    """Full key-set equality plus scale-aware per-hook comparison — a missing key
    means a rank/stage was dropped, a band miss means numerical drift."""
    assert set(cache1.keys()) == set(cache2.keys())
    for name in cache1:
        t1, t2 = cache1[name].float(), cache2[name].float()
        assert t1.shape == t2.shape, name
        assert close(t1, t2), (
            f"{name}: max abs diff {(t1 - t2).abs().max().item():.3e} "
            f"(scale {t2.abs().max().item():.1f})"
        )
