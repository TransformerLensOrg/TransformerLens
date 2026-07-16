"""TP=2 position-scoped interventions — separate file so it runs in a fresh process.

In-process vLLM engines don't reliably release GPU memory before process exit, so
this third engine boot (TP=2 + position-interventions buffers) gets its own pytest
invocation instead of stacking on test_vllm_multigpu.py's two resident engines:

    uv run pytest tests/acceptance/model_bridge/test_vllm_multigpu_pos.py -m multigpu -v
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")

from ._vllm_multigpu_common import MULTIGPU_MARKS, PROMPT_IDS, boot_multigpu

pytestmark = MULTIGPU_MARKS


@pytest.fixture(scope="module")
def pos_bridge():
    bridge = boot_multigpu(tensor_parallel_size=2, enable_position_interventions=True)
    yield bridge
    bridge.close()


class TestTPPositionInterventions:
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

    def test_pos_beyond_prompt_rejected_under_tp(self, pos_bridge):
        """The prompt-length guard must hold with spawned TP workers too."""
        with pytest.raises(ValueError, match="beyond the prompt length"):
            pos_bridge.forward(
                torch.tensor([PROMPT_IDS]),
                intervene={"embed.hook_out": {"op": "suppress", "pos": 50}},
            )
