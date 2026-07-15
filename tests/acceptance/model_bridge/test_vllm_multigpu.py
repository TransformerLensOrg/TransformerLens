"""TP=2 validation for the vLLM driver — real engine, real model, >= 2 GPUs.

Run on a multi-GPU box with the ``vllm`` extra installed:

    uv run pytest tests/acceptance/model_bridge/test_vllm_multigpu.py -m multigpu -v

Every test compares tensor_parallel_size=2 against the GPU-validated TP=1 path on
the same tiny model, so a pass means TP introduces no capture/intervention/logit
drift beyond kernel-order noise. Engines boot once per size (module fixtures) —
never boot two engines concurrently on the same devices.
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

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT_IDS = [504, 4674, 1442, 29892, 322]  # fixed ids: no tokenizer variance in scope
# vLLM kernel scheduling differs with TP (all-reduce order), so exact equality is
# not expected; band matches scripts/vllm_parity_report.py, including its
# scale-aware atol (final-layer streams reach O(1e3) and noise grows with scale).
ATOL = RTOL = 2e-2
REL_BAND = 2e-3


def _close(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    atol_eff = max(ATOL, REL_BAND * t2.abs().max().item())
    return torch.allclose(t1, t2, atol=atol_eff, rtol=RTOL)


def _boot(tp: int, **kwargs):
    from transformer_lens.model_bridge.sources.vllm.source import boot_vllm

    return boot_vllm(
        MODEL,
        dtype=torch.float32,
        max_model_len=2048,
        gpu_memory_utilization=0.35,
        tensor_parallel_size=tp,
        **kwargs,
    )


@pytest.fixture(scope="module")
def bridges():
    """TP=1 reference then TP=2 under test; serial boots, closed in reverse order."""
    b1 = _boot(1)
    b2 = _boot(2)
    yield b1, b2
    b2.close()
    b1.close()


class TestTPCaptureParity:
    def test_captures_match_tp1(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        _, cache1 = b1.run_with_cache(toks)
        _, cache2 = b2.run_with_cache(toks)
        assert set(cache1.keys()) == set(cache2.keys())
        for name in cache1:
            t1, t2 = cache1[name].float(), cache2[name].float()
            assert t1.shape == t2.shape, name
            assert _close(t1, t2), (
                f"{name}: max abs diff {(t1 - t2).abs().max().item():.3e} "
                f"(scale {t2.abs().max().item():.1f})"
            )

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

    def test_tp_replication_tripwire_ran_clean(self, bridges):
        """The first capture-bearing forward cross-checked ranks without raising."""
        _, b2 = bridges
        assert b2._driver._tp_size == 2
        assert b2._driver._tp_verified is True


class TestTPInterventionParity:
    def test_suppress_matches_tp1(self, bridges):
        b1, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        spec = {"blocks.0.mlp.hook_out": {"op": "suppress"}}
        l1 = b1.forward(toks, return_type="logits", intervene=spec)
        l2 = b2.forward(toks, return_type="logits", intervene=spec)
        assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))
        assert _close(l1.float(), l2.float())

    def test_intervention_actually_bites_under_tp(self, bridges):
        """Guard against a TP no-op passing the parity test trivially."""
        _, b2 = bridges
        toks = torch.tensor([PROMPT_IDS])
        clean = b2.forward(toks, return_type="logits")
        edited = b2.forward(
            toks, return_type="logits", intervene={"blocks.0.mlp.hook_out": {"op": "suppress"}}
        )
        assert not torch.allclose(clean.float(), edited.float(), atol=1e-4, rtol=1e-4)


class TestTPPositionInterventions:
    @pytest.fixture(scope="class")
    def pos_bridge(self):
        bridge = _boot(2, enable_position_interventions=True)
        yield bridge
        bridge.close()

    def test_pos_scoped_edit_is_row_scoped(self, pos_bridge):
        toks = torch.tensor([PROMPT_IDS])
        _, clean = pos_bridge.run_with_cache(toks)
        _, edited = pos_bridge.run_with_cache(
            toks, intervene={"embed.hook_out": {"op": "suppress", "pos": 2}}
        )
        name = "embed.hook_out"
        assert torch.allclose(
            edited[name][0, 2].float(), torch.zeros_like(edited[name][0, 2].float())
        )
        # Off-target rows untouched (row-scoped affine survived TP).
        for row in (0, 1, 3, 4):
            assert torch.allclose(
                clean[name][0, row].float(), edited[name][0, row].float(), atol=1e-5, rtol=1e-5
            )
