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

from ._vllm_multigpu_common import (
    MULTIGPU_MARKS,
    N_LAYERS,
    PROMPT_IDS,
    assert_caches_match,
    bridge_pair_fixture,
    close,
)

pytestmark = MULTIGPU_MARKS

bridges = bridge_pair_fixture(pipeline_parallel_size=2)


class TestPPCaptureParity:
    def test_captures_merge_across_stages_and_match(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        _, cache1 = b1.run_with_cache(toks)
        _, cache2 = b2.run_with_cache(toks)
        # Key-set equality doubles as the merge check: a missing later-stage hook
        # means the per-rank read or the merge dropped a stage.
        assert_caches_match(cache1, cache2)

    def test_layout_check_ran_clean(self, bridges):
        """First forward verified per-stage ownership + replica fire-counter
        agreement under PP microbatching without raising."""
        _, b2 = bridges
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
        assert close(l1.float(), l2.float())
