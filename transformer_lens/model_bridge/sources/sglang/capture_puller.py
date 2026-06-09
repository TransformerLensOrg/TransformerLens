"""Driver-side ZMQ PULL endpoint for tensors the worker hooks push.

Driver binds PULL before Engine launches (kills the connect-before-bind race);
worker connects PUSH lazily from the hook factory."""
from __future__ import annotations

from typing import Any, Dict, List


class CapturePuller:
    """Driver-side PULL socket for one Engine session. One puller per
    :class:`SGLangDriver`; channel string propagates to the worker via
    ``ServerArgs.forward_hooks``."""

    def __init__(self, channel: str) -> None:
        import zmq

        self._channel = channel
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.bind(channel)

    @property
    def channel(self) -> str:
        return self._channel

    def drain(self, timeout_ms: int = 1000) -> List[Dict[str, Any]]:
        """Read every queued message. ``timeout_ms`` bounds the wait for the
        first one; subsequent messages must arrive within ~50ms."""
        import zmq

        out: List[Dict[str, Any]] = []
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        wait = timeout_ms
        while True:
            events = dict(poller.poll(timeout=wait))
            if self._sock not in events:
                break
            try:
                msg = self._sock.recv_pyobj(zmq.NOBLOCK)
                out.append(msg)
            except zmq.Again:
                break
            wait = 50
        return out

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        # ZMQ leaves the ipc:// socket file behind on close.
        if self._channel.startswith("ipc://"):
            import os

            path = self._channel[len("ipc://") :]
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
