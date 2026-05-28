"""Unit tests for the eager batched capture hook (plugin._make_batched_hook).

The hook closure is pure torch + the segmentation helper, so it runs without a
vLLM install or GPU: a mock worker carries ``query_start_loc`` (segment_by_request's
backend-agnostic source) and the per-(req_id, hook) accumulator.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from transformer_lens.model_bridge.sources.vllm.plugin import _make_batched_hook


def _worker(specs):
    """Single-request worker mock: query_start_loc=[0, 2], one req id."""
    return SimpleNamespace(
        _tl_accum={},
        _tl_intervention_specs=specs,
        model_runner=SimpleNamespace(
            query_start_loc=torch.tensor([0, 2]),
            input_batch=SimpleNamespace(req_ids=["r0"]),
        ),
    )


class TestBatchedHookInterventionTargeting:
    """A batched hook applies ONLY the spec keyed to its own hook name."""

    def test_non_targeted_hook_is_unmodified(self):
        # Spec targets embed; a hook belonging to blocks.0.hook_out must NOT apply it.
        worker = _worker({"embed.hook_out": {"op": "suppress"}})
        counter = torch.zeros(1, dtype=torch.int64)
        hook = _make_batched_hook(worker, "blocks.0.hook_out", counter)

        out = hook(None, None, torch.ones(2, 4))
        assert torch.equal(out, torch.ones(2, 4)), "spec leaked onto a non-targeted hook"
        assert torch.equal(worker._tl_accum[("r0", "blocks.0.hook_out")][0], torch.ones(2, 4))

    def test_targeted_hook_is_modified(self):
        worker = _worker({"embed.hook_out": {"op": "suppress"}})
        counter = torch.zeros(1, dtype=torch.int64)
        hook = _make_batched_hook(worker, "embed.hook_out", counter)

        out = hook(None, None, torch.ones(2, 4))
        assert torch.equal(out, torch.zeros(2, 4)), "suppress did not apply to its target"
        assert torch.equal(worker._tl_accum[("r0", "embed.hook_out")][0], torch.zeros(2, 4))

    def test_no_spec_leaves_output_unchanged(self):
        worker = _worker({})
        counter = torch.zeros(1, dtype=torch.int64)
        hook = _make_batched_hook(worker, "embed.hook_out", counter)

        out = hook(None, None, torch.ones(2, 4) * 3)
        assert torch.equal(out, torch.ones(2, 4) * 3)
