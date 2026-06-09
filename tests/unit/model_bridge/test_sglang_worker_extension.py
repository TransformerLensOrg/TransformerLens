"""Unit tests for the SGLang worker-extension ``tl_*`` methods.

The methods get monkey-patched onto ``ModelRunner`` by ``plugin.register``; here we
exercise them as plain functions with a hand-rolled mock ``self`` carrying the
``_tl_*`` state. Mirrors the vLLM worker-extension test pattern.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.model_bridge.sources.sglang.worker_extension import (
    tl_get_param,
    tl_read_captures,
    tl_read_counter,
    tl_remove_hooks,
    tl_reset_capture_flags,
    tl_reset_counter,
    tl_set_interventions,
)


def _model_runner(width: int = 4):
    """Mock ModelRunner with the ``_tl_*`` state the methods walk."""
    hook = "blocks.0.hook_out"
    other = "blocks.1.hook_out"
    return SimpleNamespace(
        _tl_buffers={hook: torch.arange(8.0).reshape(4, 2), other: torch.zeros(4, 2)},
        _tl_scale_buffers={hook: torch.ones(width), other: torch.ones(width)},
        _tl_bias_buffers={hook: torch.zeros(width), other: torch.zeros(width)},
        _tl_capture_flags={
            hook: torch.ones(1, dtype=torch.int64),
            other: torch.ones(1, dtype=torch.int64),
        },
        _tl_fire_counter=torch.zeros(1, dtype=torch.int64),
        _tl_hook_handles=[],
        # ``self.model`` on a ModelRunner is the top-level ``ForCausalLM`` wrapper,
        # which contains the inner ``model`` (LlamaModel-equivalent). Walking
        # ``"model.embed_tokens.weight"`` therefore goes: wrapper.model → inner.embed_tokens → .weight.
        model=SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=torch.arange(6.0).reshape(2, 3))
            )
        ),
    )


class TestReadCaptures:
    def test_returns_cpu_clones_of_buffer_slices(self):
        runner = _model_runner()
        out = tl_read_captures(runner, [2])
        assert set(out) == {"blocks.0.hook_out", "blocks.1.hook_out"}
        assert out["blocks.0.hook_out"].shape == (2, 2)
        # Clone — mutating the returned tensor must not touch the buffer.
        out["blocks.0.hook_out"][0, 0] = -1.0
        assert runner._tl_buffers["blocks.0.hook_out"][0, 0] == 0.0

    def test_names_filter_restricts_read(self):
        runner = _model_runner()
        out = tl_read_captures(runner, [2], names=["blocks.0.hook_out"])
        assert set(out) == {"blocks.0.hook_out"}


class TestSetInterventions:
    def test_empty_specs_resets_to_identity(self):
        runner = _model_runner()
        runner._tl_scale_buffers["blocks.0.hook_out"].fill_(5.0)
        runner._tl_bias_buffers["blocks.0.hook_out"].fill_(7.0)
        tl_set_interventions(runner, {})
        assert torch.equal(runner._tl_scale_buffers["blocks.0.hook_out"], torch.ones(4))
        assert torch.equal(runner._tl_bias_buffers["blocks.0.hook_out"], torch.zeros(4))

    def test_suppress_zeros_scale_and_bias(self):
        runner = _model_runner()
        tl_set_interventions(runner, {"blocks.0.hook_out": {"op": "suppress"}})
        assert torch.equal(runner._tl_scale_buffers["blocks.0.hook_out"], torch.zeros(4))
        assert torch.equal(runner._tl_bias_buffers["blocks.0.hook_out"], torch.zeros(4))

    def test_scale_applies_factor(self):
        runner = _model_runner()
        tl_set_interventions(runner, {"blocks.0.hook_out": {"op": "scale", "factor": 0.5}})
        assert torch.equal(runner._tl_scale_buffers["blocks.0.hook_out"], torch.full((4,), 0.5))

    def test_add_scalar_value(self):
        runner = _model_runner()
        tl_set_interventions(runner, {"blocks.0.hook_out": {"op": "add", "value": 0.25}})
        assert torch.equal(runner._tl_bias_buffers["blocks.0.hook_out"], torch.full((4,), 0.25))
        assert torch.equal(runner._tl_scale_buffers["blocks.0.hook_out"], torch.ones(4))

    def test_set_width_vector(self):
        runner = _model_runner()
        v = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tl_set_interventions(runner, {"blocks.0.hook_out": {"op": "set", "value": v}})
        assert torch.equal(runner._tl_bias_buffers["blocks.0.hook_out"], v)
        assert torch.equal(runner._tl_scale_buffers["blocks.0.hook_out"], torch.zeros(4))

    def test_unknown_hook_raises(self):
        runner = _model_runner()
        with pytest.raises(KeyError, match="missing"):
            tl_set_interventions(runner, {"missing": {"op": "suppress"}})

    def test_invalid_value_shape_raises(self):
        runner = _model_runner()
        with pytest.raises(ValueError, match="must be a scalar or shape"):
            tl_set_interventions(
                runner,
                {"blocks.0.hook_out": {"op": "set", "value": torch.zeros(2)}},
            )

    def test_unsupported_op_raises(self):
        runner = _model_runner()
        with pytest.raises(ValueError, match="Unsupported"):
            tl_set_interventions(runner, {"blocks.0.hook_out": {"op": "rotate"}})


class TestFireCounter:
    def test_reset_and_read(self):
        runner = _model_runner()
        runner._tl_fire_counter.fill_(7)
        assert tl_read_counter(runner) == 7
        tl_reset_counter(runner)
        assert tl_read_counter(runner) == 0


class TestResetCaptureFlags:
    def test_opens_every_per_hook_gate(self):
        runner = _model_runner()
        # Confirm closed before.
        assert all(f.item() == 1 for f in runner._tl_capture_flags.values())
        tl_reset_capture_flags(runner)
        assert all(f.item() == 0 for f in runner._tl_capture_flags.values())


class TestGetParam:
    def test_dotted_path_returns_cpu_clone(self):
        runner = _model_runner()
        out = tl_get_param(runner, "model.embed_tokens.weight")
        assert out is not None and out.shape == (2, 3)
        # Mutating the clone must not mutate the original.
        out[0, 0] = -1.0
        assert runner.model.model.embed_tokens.weight[0, 0] == 0.0

    def test_missing_path_returns_none(self):
        runner = _model_runner()
        assert tl_get_param(runner, "model.nonexistent.weight") is None


class TestRemoveHooks:
    def test_idempotent_clears_buffers_and_flags(self):
        runner = _model_runner()
        runner._tl_hook_handles = [SimpleNamespace(remove=lambda: None)]
        tl_remove_hooks(runner)
        assert runner._tl_hook_handles == []
        assert runner._tl_buffers == {}
        assert runner._tl_capture_flags == {}
        # Second call no-ops cleanly.
        tl_remove_hooks(runner)
