"""Unit tests for the SGLang hook factory + Scheduler-method state mutators.

The hook closure does ZMQ ``send_pyobj`` which we mock out — the math is pure
torch so we can test the affine + materialize + capture-enabled gating without
a live socket. ZMQ round-trip is covered separately in test_sglang_capture_puller.py.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.sources.sglang import hooks


@pytest.fixture(autouse=True)
def _reset_shared_state(monkeypatch):
    """Each test starts with capture disabled + empty interventions."""
    monkeypatch.setattr(hooks, "_shared_state", {"capture_enabled": False, "interventions": {}})
    monkeypatch.setattr(hooks, "_push_sockets", {})


def _factory(monkeypatch, canonical_name="blocks.0.hook_out", materialize=False):
    """Build a hook with the PUSH socket mocked (records sends instead of sending)."""
    sent: list = []
    fake_sock = MagicMock()
    fake_sock.send_pyobj = lambda msg, flags=0: sent.append(msg)
    monkeypatch.setattr(hooks, "_get_push_socket", lambda channel: fake_sock)
    hook = hooks.make_capture_hook(
        {
            "canonical_name": canonical_name,
            "channel": "ipc:///tmp/ignored",
            "materialize": materialize,
        }
    )
    return hook, sent


class TestCaptureGating:
    def test_disabled_does_nothing(self, monkeypatch):
        hook, sent = _factory(monkeypatch)
        hook(None, None, torch.ones(2, 4))
        assert sent == []

    def test_enabled_sends_cpu_clone(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hook, sent = _factory(monkeypatch)
        hook(None, None, torch.full((2, 4), 3.0))
        assert len(sent) == 1
        msg = sent[0]
        assert msg["name"] == "blocks.0.hook_out"
        assert torch.equal(msg["tensor"], torch.full((2, 4), 3.0))


class TestInterventionPath:
    def test_suppress_zeros_and_rewraps(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hooks._shared_state["interventions"] = {"blocks.0.hook_out": {"op": "suppress"}}
        hook, sent = _factory(monkeypatch)
        out = hook(None, None, torch.ones(2, 4))
        # Returned modified tensor for downstream layers.
        assert torch.equal(out, torch.zeros(2, 4))
        # Sent captures the modified value.
        assert torch.equal(sent[0]["tensor"], torch.zeros(2, 4))

    def test_scale_factor(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hooks._shared_state["interventions"] = {"blocks.0.hook_out": {"op": "scale", "factor": 0.5}}
        hook, sent = _factory(monkeypatch)
        out = hook(None, None, torch.full((2, 4), 4.0))
        assert torch.equal(out, torch.full((2, 4), 2.0))

    def test_add_value(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hooks._shared_state["interventions"] = {"blocks.0.hook_out": {"op": "add", "value": 1.5}}
        hook, sent = _factory(monkeypatch)
        out = hook(None, None, torch.zeros(2, 4))
        assert torch.equal(out, torch.full((2, 4), 1.5))

    def test_set_width_vector(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hooks._shared_state["interventions"] = {
            "blocks.0.hook_out": {"op": "set", "value": [1.0, 2.0, 3.0, 4.0]}
        }
        hook, sent = _factory(monkeypatch)
        out = hook(None, None, torch.zeros(2, 4))
        assert torch.equal(out[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_no_intervention_returns_none(self, monkeypatch):
        """Hook returns None (PyTorch convention for "don't replace") when no spec."""
        hooks._shared_state["capture_enabled"] = True
        hook, sent = _factory(monkeypatch)
        out = hook(None, None, torch.ones(2, 4))
        assert out is None
        # But still captured.
        assert len(sent) == 1


class TestMaterializeDecoderLayer:
    def test_captures_fused_residual_sum(self, monkeypatch):
        """Decoder layer output is ``(mlp_delta, residual)``; hook captures sum."""
        hooks._shared_state["capture_enabled"] = True
        hook, sent = _factory(monkeypatch, materialize=True)
        hidden = torch.ones(2, 4)
        residual = torch.full((2, 4), 2.0)
        out = hook(None, None, (hidden, residual))
        # Captured = hidden + residual = 3.
        assert torch.equal(sent[0]["tensor"], torch.full((2, 4), 3.0))
        # No intervention → returns None (output stays the original tuple).
        assert out is None

    def test_intervention_rewraps_to_residual_tuple(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hooks._shared_state["interventions"] = {"blocks.0.hook_out": {"op": "suppress"}}
        hook, sent = _factory(monkeypatch, materialize=True)
        hidden = torch.ones(2, 4)
        residual = torch.full((2, 4), 2.0)
        out = hook(None, None, (hidden, residual))
        # Modified = 0; rewrapped = (modified - residual, residual) = (-2, 2).
        new_hidden, new_residual = out
        assert torch.equal(new_hidden, torch.full((2, 4), -2.0))
        assert torch.equal(new_residual, residual)


class TestSchedulerMethods:
    """The methods :func:`worker_plugin.register` patches onto Scheduler — invoked
    via ``engine.collective_rpc(...)`` in the worker subprocess."""

    def test_tl_set_interventions_replaces_state(self):
        hooks.tl_set_interventions(None, {"blocks.0.hook_out": {"op": "suppress"}})
        assert hooks._shared_state["interventions"] == {"blocks.0.hook_out": {"op": "suppress"}}

    def test_tl_set_interventions_empty_clears(self):
        hooks._shared_state["interventions"] = {"x": {"op": "suppress"}}
        hooks.tl_set_interventions(None, {})
        assert hooks._shared_state["interventions"] == {}

    def test_tl_set_capture_enabled_toggle(self):
        hooks.tl_set_capture_enabled(None, True)
        assert hooks._shared_state["capture_enabled"] is True
        hooks.tl_set_capture_enabled(None, False)
        assert hooks._shared_state["capture_enabled"] is False

    def test_tl_clear_state_resets_both(self):
        hooks._shared_state["interventions"] = {"x": {"op": "suppress"}}
        hooks._shared_state["capture_enabled"] = True
        hooks.tl_clear_state(None)
        assert hooks._shared_state["interventions"] == {}
        assert hooks._shared_state["capture_enabled"] is False

    def test_scheduler_method_names_match_module_functions(self):
        """The names worker_plugin.register iterates must resolve to functions."""
        for name in hooks.SCHEDULER_METHODS:
            assert callable(getattr(hooks, name))


class TestGetParam:
    """``tl_get_param`` walks Scheduler→tp_worker→ModelRunner→model, pushes the
    named tensor on the channel as ``{"_param": ..., "tensor": ...}``."""

    def _scheduler_with_weight(self, dotted_path: str, tensor: torch.Tensor):
        """Mock Scheduler whose tp_workers[0].model_runner.model holds a weight
        at ``dotted_path``."""
        parts = dotted_path.split(".")
        # Build a nested SimpleNamespace tree from the inside out.
        leaf = tensor
        node: Any = SimpleNamespace(**{parts[-1]: leaf})
        for seg in reversed(parts[:-1]):
            node = SimpleNamespace(**{seg: node})
        model_runner = SimpleNamespace(model=node)
        tp_worker = SimpleNamespace(model_runner=model_runner)
        return SimpleNamespace(tp_workers=[tp_worker])

    def test_pushes_weight_on_channel(self, monkeypatch):
        sent: list = []
        fake_sock = MagicMock()
        fake_sock.send_pyobj = lambda msg, flags=0: sent.append(msg)
        monkeypatch.setattr(hooks, "_get_push_socket", lambda channel: fake_sock)

        weight = torch.arange(8.0).reshape(2, 4)
        scheduler = self._scheduler_with_weight("embed_tokens.weight", weight)
        hooks.tl_get_param(scheduler, "embed_tokens.weight", "ipc:///tmp/x")

        assert len(sent) == 1
        assert sent[0]["_param"] == "embed_tokens.weight"
        assert torch.equal(sent[0]["tensor"], weight)

    def test_missing_path_silently_returns(self, monkeypatch):
        sent: list = []
        monkeypatch.setattr(
            hooks,
            "_get_push_socket",
            lambda channel: MagicMock(send_pyobj=lambda msg, flags=0: sent.append(msg)),
        )
        scheduler = SimpleNamespace(tp_workers=[])  # no worker → no model
        hooks.tl_get_param(scheduler, "anything", "ipc:///tmp/x")
        assert sent == []

    def test_non_tensor_target_silently_returns(self, monkeypatch):
        sent: list = []
        monkeypatch.setattr(
            hooks,
            "_get_push_socket",
            lambda channel: MagicMock(send_pyobj=lambda msg, flags=0: sent.append(msg)),
        )
        # weight points at a Python int, not a tensor.
        scheduler = self._scheduler_with_weight("config.value", 42)  # type: ignore[arg-type]
        hooks.tl_get_param(scheduler, "config.value", "ipc:///tmp/x")
        assert sent == []


class TestApplyInterventionInline:
    """The pure affine function the hook calls."""

    def test_unsupported_op_silent_passthrough(self):
        # Validation happens driver-side; hook is permissive on unknown ops.
        t = torch.ones(2, 4)
        out = hooks._apply_intervention_inline(t, {"op": "rotate"})
        assert torch.equal(out, t)

    def test_set_with_scalar_broadcasts(self):
        out = hooks._apply_intervention_inline(torch.zeros(2, 4), {"op": "set", "value": 7.0})
        assert torch.equal(out, torch.full((2, 4), 7.0))


class TestOutputShapePreservation:
    """Hook must preserve the output shape and tuple structure for downstream layers."""

    def test_tuple_with_tail_preserves_tail(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hooks._shared_state["interventions"] = {"blocks.0.hook_out": {"op": "suppress"}}
        hook, _ = _factory(monkeypatch)
        meta = SimpleNamespace(info="extra")
        out = hook(None, None, (torch.ones(2, 4), meta))
        assert isinstance(out, tuple) and out[1] is meta
        assert torch.equal(out[0], torch.zeros(2, 4))

    def test_non_tensor_output_ignored(self, monkeypatch):
        hooks._shared_state["capture_enabled"] = True
        hook, sent = _factory(monkeypatch)
        out = hook(None, None, "not a tensor")
        assert out is None
        assert sent == []
