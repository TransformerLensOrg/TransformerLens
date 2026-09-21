"""TP=2 validation for the vLLM driver — real engine, real model, >= 2 GPUs.

Run on a multi-GPU box with the ``vllm`` extra installed:

    uv run pytest tests/acceptance/model_bridge/test_vllm_multigpu.py -m multigpu -v

Every test compares tensor_parallel_size=2 against the GPU-validated TP=1 path on
the same tiny model, so a pass means TP introduces no capture/intervention/logit
drift beyond kernel-order noise. Engines boot once per size (module fixtures);
position-scoped interventions run separately (test_vllm_multigpu_pos.py) so the
third engine gets a fresh process.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")

from ._vllm_multigpu_common import (
    MULTIGPU_MARKS,
    PROMPT_IDS,
    assert_caches_match,
    bridge_pair_fixture,
    close,
)

pytestmark = MULTIGPU_MARKS

bridges = bridge_pair_fixture(tensor_parallel_size=2)


class TestTPCaptureParity:
    def test_captures_match_tp1(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        _, cache1 = b1.run_with_cache(toks)
        _, cache2 = b2.run_with_cache(toks)
        assert_caches_match(cache1, cache2)

    def test_logits_argmax_matches_tp1(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        l1 = b1.forward(toks, return_type="logits")
        l2 = b2.forward(toks, return_type="logits")
        assert l1.shape == l2.shape
        # Full-sequence argmax agreement — exercises the sharded-lm_head gather.
        assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))

    def test_sequence_logits_and_loss_under_tp(self, bridges):
        _, b2 = bridges
        assert b2._driver.provides_sequence_logits is True  # gather succeeded
        loss = b2.forward(torch.tensor([PROMPT_IDS]), return_type="loss")
        assert torch.isfinite(loss)

    def test_layout_check_ran_clean(self, bridges):
        """The first capture-bearing forward cross-checked ranks without raising."""
        _, b2 = bridges
        assert b2._driver._tp_size == 2
        assert b2._driver._layout_verified is True


class TestTPInterventionParity:
    def test_suppress_matches_tp1(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        spec = {"blocks.0.mlp.hook_out": {"op": "suppress"}}
        l1 = b1.forward(toks, return_type="logits", intervene=spec)
        l2 = b2.forward(toks, return_type="logits", intervene=spec)
        assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))
        assert close(l1.float(), l2.float())

    def test_intervention_actually_bites_under_tp(self, bridges):
        """Guard against a TP no-op passing the parity test trivially."""
        _, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        clean = b2.forward(toks, return_type="logits")
        edited = b2.forward(
            toks, return_type="logits", intervene={"blocks.0.mlp.hook_out": {"op": "suppress"}}
        )
        assert not torch.allclose(clean.float(), edited.float(), atol=1e-4, rtol=1e-4)


# Position-scoped interventions live in test_vllm_multigpu_pos.py — a third
# engine boot gets its own process (in-process engines under-release GPU memory).
