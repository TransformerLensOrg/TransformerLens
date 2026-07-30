"""Availability-gated real-weight integration tests for the SSM / recurrent families.

The from_config tiny tests (test_mamba_adapter / test_mamba2_adapter /
test_nemotron_h_tiny / test_granite_moe_hybrid_adapter and the Qwen3_5 unit tests)
exercise the plumbing on random init. These load *real trained checkpoints* and run
the same interp surface — family-agnostic SSM-layer discovery, recurrent-state
reconstruction, effective attention, and the opt-in eager-scan intervention path —
where the numerics differ from random init.

Every model is loaded through the ``bridge`` fixture, which SKIPS (never fails) when
the checkpoint or network is unavailable, the installed transformers lacks the
architecture, or there isn't enough memory. Point any family at a locally-cached
checkpoint with its env var (e.g. ``TL_MAMBA1_MODEL``); families without a small
official checkpoint (NemotronH 8B, Qwen3-Next 80B, …) skip in normal environments.

All ``slow`` so ``make integration-test`` (``-m "not slow"``) never downloads.
"""
import gc
import os
from dataclasses import dataclass

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class _Case:
    label: str
    env: str
    default: str  # canonical HF id, or "" for env-only (no small official checkpoint)


# Ordered by how likely the checkpoint is small enough to actually run in CI.
CASES = [
    _Case("mamba1", "TL_MAMBA1_MODEL", "state-spaces/mamba-130m-hf"),
    _Case("mamba2", "TL_MAMBA2_MODEL", "AntonV/mamba2-130m-hf"),
    _Case("granite", "TL_GRANITE_MODEL", "ibm-granite/granite-4.0-tiny-preview"),
    _Case("nemotron_h", "TL_NEMOTRON_H_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
    _Case("qwen3_next", "TL_QWEN3_NEXT_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
    _Case("qwen3_5", "TL_QWEN3_5_MODEL", ""),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c.label)
def bridge(request):
    """Load one family's real checkpoint, or skip if it can't be obtained."""
    case: _Case = request.param
    model_id = os.environ.get(case.env, case.default)
    if not model_id:
        pytest.skip(f"no default checkpoint for {case.label}; set {case.env} to run")
    try:
        # float32 (not bf16) so the reconstruction-faithfulness tolerances hold.
        br = TransformerBridge.boot_transformers(model_id, device=_device(), dtype=torch.float32)
    except pytest.skip.Exception:
        raise
    except Exception as e:  # availability gate: network / missing arch / OOM / gated repo
        pytest.skip(f"{case.label} checkpoint {model_id!r} unavailable: {type(e).__name__}: {e}")
    yield br
    del br
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="module")
def tokens():
    # Small ids valid for every real vocab; short seq keeps the O(seq^2)/O(seq) work cheap.
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])


@pytest.fixture(scope="module")
def cache(bridge, tokens):
    # use_cache=False forces the hooked prefill path so gated-delta-net interior hooks fire.
    with torch.no_grad():
        _, c = bridge.run_with_cache(tokens.to(_device()), use_cache=False)
    return c


def _realized_ssm_mixers(bridge):
    """The realized SSM mixers (find_ssm_mixer excludes hybrid passthrough slots)."""
    from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
        find_ssm_mixer,
    )

    out = []
    for block in bridge.blocks:
        mixer = find_ssm_mixer(block)
        if mixer is not None:
            out.append(mixer)
    return out


class TestRealWeightSSMSurface:
    def test_ssm_layers_nonempty(self, cache):
        layers = cache.ssm_layers()
        assert layers == sorted(layers)
        assert len(layers) > 0, "a recurrent model must expose at least one SSM layer"

    def test_compute_ssm_state_finite(self, cache, tokens):
        state = cache.compute_ssm_state()
        mats = (
            list(state.values())
            if isinstance(state, dict)
            else [state[i] for i in range(state.shape[0])]
        )
        assert len(mats) == len(cache.ssm_layers())
        for S in mats:
            assert torch.isfinite(S).all()
            # batch dim leads; a seq axis of the right length is present somewhere.
            assert S.shape[0] == tokens.shape[0]
            assert tokens.shape[1] in tuple(S.shape)

    def test_effective_attention_finite_causal(self, cache, tokens):
        M = cache.compute_ssm_effective_attention()
        mats = list(M.values()) if isinstance(M, dict) else [M[i] for i in range(M.shape[0])]
        seq = tokens.shape[1]
        upper = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        for m in mats:
            assert m.shape[-1] == m.shape[-2] == seq
            assert torch.isfinite(m).all()
            assert torch.all(m[..., upper] == 0), "effective attention must be causal"

    def test_eager_scan_matches_fused(self, bridge, tokens):
        """Reimplemented eager recurrence must reproduce the fused-kernel logits.

        Enables eager_scan on every realized SSM mixer (Mamba-1/2, gated-delta-net,
        and the SSM2-wired hybrid mixers) and compares to the default fused path on
        real trained weights — the end-to-end faithfulness gate for the intervention
        surface.
        """
        mixers = [m for m in _realized_ssm_mixers(bridge) if hasattr(type(m), "eager_scan")]
        if not mixers:
            pytest.skip("no eager-scan-capable SSM mixers in this model")

        toks = tokens.to(_device())
        with torch.no_grad():
            fused = bridge(toks, use_cache=False)
            for m in mixers:
                m.eager_scan = True
            try:
                eager = bridge(toks, use_cache=False)
            finally:
                for m in mixers:
                    m.eager_scan = False
        rel = (eager.float() - fused.float()).abs().max().item() / max(
            fused.float().abs().max().item(), 1e-8
        )
        assert rel < 2e-2, f"eager vs fused logit parity rel diff {rel:.2e}"
