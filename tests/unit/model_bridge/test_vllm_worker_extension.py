"""Unit tests for TLWorkerExtension — the post-compile read/write RPC surface.

These methods are pure torch + Python (no vLLM install needed). Hook installation
and the in-compile counter increment live in plugin.py and can only be exercised
on a real GPU via demos/vLLM_Bridge_Integration_Test.ipynb.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from transformer_lens.model_bridge.sources.vllm.worker_extension import (
    TLWorkerExtension,
    _apply_intervention,
    _apply_op,
)


class TestGetParam:
    """tl_get_param resolves a dotted path to a CPU tensor clone, or None."""

    def test_reads_named_tensor(self):
        ext = TLWorkerExtension()
        weight = torch.ones(4)
        ext.model_runner = SimpleNamespace(  # type: ignore[attr-defined]
            model=SimpleNamespace(norm=SimpleNamespace(weight=weight))
        )
        out = ext.tl_get_param("norm.weight")
        assert torch.equal(out, weight)
        assert out is not weight  # cloned, not a live reference

    def test_missing_path_returns_none(self):
        ext = TLWorkerExtension()
        ext.model_runner = SimpleNamespace(model=SimpleNamespace())  # type: ignore[attr-defined]
        assert ext.tl_get_param("norm.weight") is None

    def test_non_tensor_target_returns_none(self):
        ext = TLWorkerExtension()
        ext.model_runner = SimpleNamespace(  # type: ignore[attr-defined]
            model=SimpleNamespace(norm=SimpleNamespace())
        )
        assert ext.tl_get_param("norm") is None


class TestReadCapturesFiltering:
    """names restricts the GPU→CPU read; None reads all."""

    def _ext(self):
        ext = TLWorkerExtension()
        ext._tl_buffers = {"a": torch.ones(3, 4), "b": torch.zeros(3, 4)}
        return ext

    def test_names_filters(self):
        out = self._ext().tl_read_captures([2], names=["a"])
        assert set(out) == {"a"}
        assert tuple(out["a"].shape) == (2, 4)

    def test_none_reads_all(self):
        out = self._ext().tl_read_captures([2])
        assert set(out) == {"a", "b"}

    def test_batched_names_filters(self):
        ext = TLWorkerExtension()
        ext._tl_accum = {
            ("r", "a"): [torch.ones(2, 4)],
            ("r", "b"): [torch.zeros(2, 4)],
        }
        out = ext.tl_read_batched_captures(names=["a"])
        assert set(out["r"]) == {"a"}


class TestFireCounter:
    """tl_reset_counter / tl_read_counter operate on the shared GPU counter."""

    def test_read_counter_reflects_value(self):
        ext = TLWorkerExtension()
        ext._tl_fire_counter = torch.tensor([7], dtype=torch.int64)
        assert ext.tl_read_counter() == 7

    def test_reset_counter_zeros(self):
        ext = TLWorkerExtension()
        ext._tl_fire_counter = torch.tensor([42], dtype=torch.int64)
        ext.tl_reset_counter()
        assert ext.tl_read_counter() == 0

    def test_missing_counter_reads_zero(self):
        """Before patched_load_model runs, the attribute doesn't exist yet."""
        ext = TLWorkerExtension()
        assert ext.tl_read_counter() == 0
        ext.tl_reset_counter()  # no-op, must not raise


class TestApplyIntervention:
    """_apply_intervention translates spec dicts to affine buffer writes."""

    def _buffers(self, width=4):
        return torch.ones(width), torch.zeros(width)

    def test_suppress_zeros_both(self):
        scale, bias = self._buffers()
        _apply_intervention(scale, bias, {"op": "suppress"})
        assert torch.equal(scale, torch.zeros(4))
        assert torch.equal(bias, torch.zeros(4))

    def test_scale_sets_factor(self):
        scale, bias = self._buffers()
        _apply_intervention(scale, bias, {"op": "scale", "factor": 0.5})
        assert torch.equal(scale, torch.full((4,), 0.5))
        assert torch.equal(bias, torch.zeros(4))

    def test_add_scalar_broadcasts(self):
        scale, bias = self._buffers()
        _apply_intervention(scale, bias, {"op": "add", "value": 0.5})
        assert torch.equal(scale, torch.ones(4))
        assert torch.equal(bias, torch.full((4,), 0.5))

    def test_add_vector_elementwise(self):
        scale, bias = self._buffers()
        vec = [1.0, 2.0, 3.0, 4.0]
        _apply_intervention(scale, bias, {"op": "add", "value": vec})
        assert torch.equal(bias, torch.tensor(vec))

    def test_set_zeros_scale(self):
        scale, bias = self._buffers()
        _apply_intervention(scale, bias, {"op": "set", "value": 2.0})
        assert torch.equal(scale, torch.zeros(4))
        assert torch.equal(bias, torch.full((4,), 2.0))

    def test_value_shape_mismatch_raises(self):
        import pytest

        scale, bias = self._buffers(width=4)
        with pytest.raises(ValueError, match="must be a scalar or shape"):
            _apply_intervention(scale, bias, {"op": "add", "value": [1.0, 2.0]})


class TestApplyOp:
    """_apply_op is the eager batched path's tensor-level intervention."""

    def _t(self):
        return torch.ones(3, 4)

    def test_suppress_zeros(self):
        assert torch.equal(_apply_op(self._t(), {"op": "suppress"}), torch.zeros(3, 4))

    def test_scale_multiplies(self):
        assert torch.equal(
            _apply_op(self._t(), {"op": "scale", "factor": 0.5}), torch.full((3, 4), 0.5)
        )

    def test_add_scalar_broadcasts(self):
        assert torch.equal(
            _apply_op(self._t(), {"op": "add", "value": 1.0}), torch.full((3, 4), 2.0)
        )

    def test_add_vector_elementwise(self):
        out = _apply_op(self._t(), {"op": "add", "value": [0.0, 1.0, 2.0, 3.0]})
        assert torch.equal(out[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_set_replaces(self):
        assert torch.equal(
            _apply_op(self._t(), {"op": "set", "value": 9.0}), torch.full((3, 4), 9.0)
        )

    def test_unsupported_op_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unsupported intervention op"):
            _apply_op(self._t(), {"op": "clamp", "value": 1.0})

    def test_value_shape_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError, match="must be a scalar or shape"):
            _apply_op(self._t(), {"op": "add", "value": [1.0, 2.0]})


class TestBatchedAccumulators:
    """tl_reset_accumulators / tl_read_batched_captures / tl_set_batched_interventions."""

    def test_read_concatenates_chunks_in_order(self):
        ext = TLWorkerExtension()
        ext._tl_accum = {
            ("req-A", "embed.hook_out"): [torch.ones(2, 4), torch.full((1, 4), 2.0)],
            ("req-B", "embed.hook_out"): [torch.full((3, 4), 5.0)],
        }
        out = ext.tl_read_batched_captures()
        # req-A's two chunks cat to (3, 4); chunk order is token order.
        assert tuple(out["req-A"]["embed.hook_out"].shape) == (3, 4)
        assert torch.equal(out["req-A"]["embed.hook_out"][2], torch.full((4,), 2.0))
        assert tuple(out["req-B"]["embed.hook_out"].shape) == (3, 4)

    def test_reset_clears_accumulator(self):
        ext = TLWorkerExtension()
        ext._tl_accum = {("r", "h"): [torch.ones(1, 4)]}
        ext.tl_reset_accumulators()
        assert ext.tl_read_batched_captures() == {}

    def test_set_batched_interventions_validates_op(self):
        import pytest

        ext = TLWorkerExtension()
        with pytest.raises(ValueError, match="Unsupported intervention op"):
            ext.tl_set_batched_interventions({"h": {"op": "clamp"}})

    def test_set_batched_interventions_stores_specs(self):
        ext = TLWorkerExtension()
        ext.tl_set_batched_interventions({"h": {"op": "suppress"}})
        assert ext._tl_intervention_specs == {"h": {"op": "suppress"}}
