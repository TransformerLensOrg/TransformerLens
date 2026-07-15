"""PP=2 validation for the vLLM driver — real engine, real model, >= 2 GPUs.

Run on a multi-GPU box with the ``vllm`` extra installed (own process — the
sibling TP suite boots its own engines):

    uv run pytest tests/acceptance/model_bridge/test_vllm_multigpu_pp.py -m multigpu -v

Compares pipeline_parallel_size=2 against the GPU-validated single-rank path:
captures must merge across stages with no gaps, interventions must bite on
hooks owned by BOTH stages, and the fire-counter layout check must hold under
vLLM's PP microbatching.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")

pytestmark = [
    pytest.mark.multigpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="needs >= 2 CUDA devices",
    ),
]

# Qwen2.5-0.5B: 24 layers -> 12 per PP stage; even-headed (irrelevant for PP but
# keeps the model shared with the TP suite).
MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT_IDS = [504, 4674, 1442, 29892, 322]
ATOL = RTOL = 2e-2
REL_BAND = 2e-3
N_LAYERS = 24


def _close(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    atol_eff = max(ATOL, REL_BAND * t2.abs().max().item())
    return torch.allclose(t1, t2, atol=atol_eff, rtol=RTOL)


def _boot(pp: int):
    from transformer_lens.model_bridge.sources.vllm.source import boot_vllm

    return boot_vllm(
        MODEL,
        dtype=torch.float32,
        max_model_len=2048,
        gpu_memory_utilization=0.35,
        pipeline_parallel_size=pp,
    )


@pytest.fixture(scope="module")
def bridges():
    """Single-rank reference then PP=2 under test; serial boots, closed in reverse."""
    b1 = _boot(1)
    b2 = _boot(2)
    yield b1, b2
    b2.close()
    b1.close()


class TestPPCaptureParity:
    def test_captures_merge_across_stages_and_match(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        _, cache1 = b1.run_with_cache(toks)
        _, cache2 = b2.run_with_cache(toks)
        # The merge must reassemble the FULL hook set — a missing later-stage hook
        # means the per-rank read or the merge dropped a stage.
        assert set(cache1.keys()) == set(cache2.keys())
        for name in cache1:
            t1, t2 = cache1[name].float(), cache2[name].float()
            assert t1.shape == t2.shape, name
            assert _close(t1, t2), (
                f"{name}: max abs diff {(t1 - t2).abs().max().item():.3e} "
                f"(scale {t2.abs().max().item():.1f})"
            )

    def test_layout_check_ran_clean(self, bridges):
        """First forward verified per-stage ownership + counter proportionality
        under PP microbatching without raising."""
        _, b2 = bridges
        assert b2._driver._pp_size == 2
        assert b2._driver._layout_verified is True

    def test_logits_argmax_matches(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        l1 = b1.forward(toks, return_type="logits")
        l2 = b2.forward(toks, return_type="logits")
        # Exercises the stage-local gathers: lm_head + final norm live on stage 1.
        assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))

    def test_sequence_logits_and_loss_under_pp(self, bridges):
        _, b2 = bridges
        assert b2._driver.provides_sequence_logits is True
        loss = b2.forward(torch.tensor([PROMPT_IDS]), return_type="loss")
        assert torch.isfinite(loss)


class TestPPInterventionParity:
    @pytest.mark.parametrize(
        "hook",
        [
            "blocks.0.mlp.hook_out",  # first stage
            f"blocks.{N_LAYERS - 1}.mlp.hook_out",  # last stage
        ],
    )
    def test_suppress_bites_and_matches_single_rank(self, bridges, hook):
        """Interventions must apply on whichever stage owns the hook."""
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        spec = {hook: {"op": "suppress"}}
        clean2 = b2.forward(toks, return_type="logits")
        l1 = b1.forward(toks, return_type="logits", intervene=spec)
        l2 = b2.forward(toks, return_type="logits", intervene=spec)
        assert not torch.allclose(
            clean2.float(), l2.float(), atol=1e-4, rtol=1e-4
        ), f"{hook}: intervention was a no-op under PP"
        assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))
        assert _close(l1.float(), l2.float())
