"""Unit tests for :class:`CapturePuller` — exercise a real ZMQ PUSH/PULL pair.

Skips when ``pyzmq`` isn't installed. Uses a per-test ``ipc://`` channel under
``tmp_path`` so concurrent tests don't collide.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("zmq")

from transformer_lens.model_bridge.sources.sglang.capture_puller import (  # noqa: E402
    CapturePuller,
)


@pytest.fixture
def channel(tmp_path):
    return f"ipc://{tmp_path / 'tl_test.sock'}"


def _push_socket(channel):
    """Build a PUSH socket connected to the driver-bound PULL."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(channel)
    return sock


class TestRoundTrip:
    def test_single_message_round_trip(self, channel):
        puller = CapturePuller(channel)
        try:
            push = _push_socket(channel)
            push.send_pyobj({"name": "blocks.0.hook_out", "tensor": torch.ones(2, 4)})
            msgs = puller.drain(timeout_ms=1000)
            assert len(msgs) == 1
            assert msgs[0]["name"] == "blocks.0.hook_out"
            assert torch.equal(msgs[0]["tensor"], torch.ones(2, 4))
        finally:
            puller.close()

    def test_multiple_messages_drained_in_order(self, channel):
        puller = CapturePuller(channel)
        try:
            push = _push_socket(channel)
            for i in range(5):
                push.send_pyobj({"name": f"hook_{i}", "tensor": torch.full((2,), float(i))})
            msgs = puller.drain(timeout_ms=1000)
            assert [m["name"] for m in msgs] == [f"hook_{i}" for i in range(5)]
        finally:
            puller.close()

    def test_empty_socket_returns_empty_list(self, channel):
        puller = CapturePuller(channel)
        try:
            assert puller.drain(timeout_ms=50) == []
        finally:
            puller.close()

    def test_drain_after_close_is_safe(self, channel):
        puller = CapturePuller(channel)
        puller.close()
        # Re-close is a no-op (close() catches the error).
        puller.close()

    def test_close_unlinks_ipc_socket_file(self, channel):
        import os

        path = channel[len("ipc://") :]
        puller = CapturePuller(channel)
        # Bind creates the socket file on disk.
        assert os.path.exists(path)
        puller.close()
        # close() unlinks it — protects /tmp from accumulating stale .sock corpses.
        assert not os.path.exists(path)
