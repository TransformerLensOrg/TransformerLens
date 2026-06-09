"""Unit tests for the SGLang compiled-mode capture hook (``plugin._make_capture_hook``).

The hook closure is pure torch (mirrors the vLLM compiled-mode hook); no SGLang
install or GPU needed. Tests pin first-write-wins, the materialize-tuple decoder-layer
path, and the affine-intervention pass-through.
"""
from __future__ import annotations

import torch

from transformer_lens.model_bridge.sources.sglang.plugin import (
    _gated_capture,
    _make_capture_hook,
)


def _buffers(max_n: int = 4, width: int = 8):
    capture = torch.zeros(max_n, width)
    scale = torch.ones(width)
    bias = torch.zeros(width)
    counter = torch.zeros(1, dtype=torch.int64)
    flag = torch.zeros(1, dtype=torch.int64)  # open
    return capture, scale, bias, counter, flag


class TestGatedCapture:
    """First-write-wins via torch.where — open writes, closed self-copies."""

    def test_open_flag_writes(self):
        cap = torch.zeros(4, 2)
        flag = torch.zeros(1, dtype=torch.int64)
        modified = torch.ones(2, 2) * 5.0
        _gated_capture(cap, 2, modified, flag)
        assert torch.equal(cap[:2], torch.ones(2, 2) * 5.0)
        assert flag.item() == 1  # closed after write

    def test_closed_flag_self_copies(self):
        cap = torch.full((4, 2), 9.0)
        flag = torch.ones(1, dtype=torch.int64)  # closed
        modified = torch.zeros(2, 2)
        _gated_capture(cap, 2, modified, flag)
        # cap[:2] unchanged from the prior write.
        assert torch.equal(cap[:2], torch.full((2, 2), 9.0))
        assert flag.item() == 1  # still closed


class TestCaptureHookFlat:
    """Default (materialize=False) path: tensor or tuple output."""

    def test_writes_and_returns_modified_tensor(self):
        capture, scale, bias, counter, flag = _buffers()
        hook = _make_capture_hook(capture, scale, bias, counter, flag, materialize=False)

        t = torch.ones(2, 8) * 3.0
        out = hook(None, None, t)
        assert torch.equal(out, t)  # identity scale/bias
        assert torch.equal(capture[:2], t)
        assert counter.item() == 1
        assert flag.item() == 1  # closed after first write

    def test_intervention_affine_applied_and_captured(self):
        capture, scale, bias, counter, flag = _buffers()
        scale.fill_(2.0)
        bias.fill_(1.0)
        hook = _make_capture_hook(capture, scale, bias, counter, flag, materialize=False)

        t = torch.ones(2, 8)
        out = hook(None, None, t)
        # modified = t * 2 + 1 = 3
        assert torch.equal(out, torch.full((2, 8), 3.0))
        # capture mirrors the modified value.
        assert torch.equal(capture[:2], torch.full((2, 8), 3.0))

    def test_tuple_output_preserves_tail(self):
        capture, scale, bias, counter, flag = _buffers()
        hook = _make_capture_hook(capture, scale, bias, counter, flag, materialize=False)

        t = torch.ones(2, 8) * 7.0
        meta = {"info": "extra"}
        out = hook(None, None, (t, meta))
        assert isinstance(out, tuple) and out[0].shape == (2, 8) and out[1] == meta


class TestCaptureHookMaterialize:
    """Decoder-layer path: output is ``(mlp_delta, residual)``; capture the sum."""

    def test_captures_fused_residual_sum(self):
        capture, scale, bias, counter, flag = _buffers(max_n=2, width=4)
        hook = _make_capture_hook(capture, scale, bias, counter, flag, materialize=True)

        hidden = torch.ones(2, 4)  # mlp_delta
        residual = torch.full((2, 4), 2.0)  # residual stream so far
        out = hook(None, None, (hidden, residual))

        # Captured = mlp_delta + residual = 3 everywhere.
        assert torch.equal(capture[:2], torch.full((2, 4), 3.0))
        # Returned tuple reconstructs the fused sum in the next layer's norm:
        # (modified - residual, residual). At identity, modified == hidden + residual,
        # so modified - residual == hidden.
        new_hidden, new_residual = out
        assert torch.equal(new_hidden, hidden)
        assert torch.equal(new_residual, residual)


class TestFireCounter:
    """Counter increments every call — surfaces hook double-fire under compile."""

    def test_counter_increments_per_call(self):
        capture, scale, bias, counter, flag = _buffers()
        hook = _make_capture_hook(capture, scale, bias, counter, flag, materialize=False)
        for _ in range(5):
            hook(None, None, torch.ones(2, 8))
        assert counter.item() == 5
