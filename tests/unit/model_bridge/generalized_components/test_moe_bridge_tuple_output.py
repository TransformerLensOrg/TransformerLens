"""Tests for MoEBridge tuple-output validation and router-score hooks.

Unlike the pretrain-adapter's TestDenseOrMoEFeedForwardBridgeTupleOutput
(which patches _delegate.forward and therefore exercises
DenseOrMoEFeedForwardBridge.forward in isolation), these tests call
MoEBridge.forward() directly via a stub original_component, so they cover
MoEBridge's own tuple-validation contract rather than the dispatcher's.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components import MoEBridge


class _StubMoEModule(nn.Module):
    """Minimal nn.Module whose forward is swapped in per-test, standing in
    for a real MoE layer wrapped by MoEBridge."""

    def __init__(self, fake_forward):
        super().__init__()
        # Give the module at least one parameter so
        # `next(self.parameters())` in MoEBridge.forward doesn't raise
        # StopIteration and skip the dtype-cast branch entirely.
        self.dummy = nn.Linear(4, 4)
        self._fake_forward = fake_forward

    def forward(self, *args, **kwargs):
        return self._fake_forward(*args, **kwargs)


def _bridge_with_stub(fake_forward) -> MoEBridge:
    bridge = MoEBridge(name="mlp")
    bridge.set_original_component(_StubMoEModule(fake_forward))
    return bridge


class TestMoEBridgeTupleOutput:
    def test_empty_tuple_raises_clear_type_error(self) -> None:
        bridge = _bridge_with_stub(lambda *a, **kw: ())
        with pytest.raises(TypeError, match="torch.Tensor"):
            bridge(torch.ones(1, 3, 4))

    def test_non_tensor_metadata_preserved_without_router_hook(self) -> None:
        bridge = _bridge_with_stub(lambda *a, **kw: (torch.zeros(1, 3, 4), "not_router_scores"))
        fired = {"called": False}

        def mark_fired(value, hook):
            fired["called"] = True

        bridge.hook_router_scores.add_hook(mark_fired)

        output = bridge(torch.ones(1, 3, 4))

        assert isinstance(output, tuple)
        assert output[1] == "not_router_scores"
        torch.testing.assert_close(output[0], torch.zeros(1, 3, 4))
        assert fired["called"] is False

    def test_tensor_router_scores_still_fire_hook(self) -> None:
        router_scores = torch.rand(1, 3, 8)
        bridge = _bridge_with_stub(lambda *a, **kw: (torch.zeros(1, 3, 4), router_scores))
        captured = {}

        def capture_router_scores(value, hook):
            captured["router_scores"] = value
            return value

        bridge.hook_router_scores.add_hook(capture_router_scores)

        output = bridge(torch.ones(1, 3, 4))

        assert isinstance(output, tuple)
        torch.testing.assert_close(output[1], router_scores)
        assert "router_scores" in captured
        torch.testing.assert_close(captured["router_scores"], router_scores)

    def test_non_tensor_first_element_raises_clear_type_error(self) -> None:
        """Exercises the output[0]-validation added to MoEBridge itself
        (not just the dispatcher) -- a malformed first element should
        raise a clear TypeError here rather than failing inside
        HookPoint's own type checking."""
        bridge = _bridge_with_stub(lambda *a, **kw: ("not_a_tensor", torch.zeros(1, 3, 4)))
        with pytest.raises(TypeError, match="torch.Tensor"):
            bridge(torch.ones(1, 3, 4))
