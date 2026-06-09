"""Forward-hook factory called by SGLang's ``register_forward_hooks`` in the worker
subprocess + the Scheduler methods :mod:`worker_plugin` installs to mutate the
factory's shared state from the driver via ``engine.collective_rpc``."""
from __future__ import annotations

from typing import Any, Dict

import torch

from .intervention_specs import SUPPORTED_OPS

# Read by hook closures every forward; written by the tl_* Scheduler methods below.
_shared_state: Dict[str, Any] = {
    "capture_enabled": False,
    "interventions": {},  # canonical_name → {"op": ..., **params}
}

# Single PUSH socket per channel, shared by every hook bound to the same channel.
_push_sockets: Dict[str, Any] = {}


def make_capture_hook(config: Dict[str, Any]):
    """``register_forward_hook`` callable for one canonical TL hook.

    ``config`` keys: ``canonical_name`` (wire key), ``channel`` (``ipc://`` PULL
    address), ``materialize`` (True for decoder layers — output is the fused-residual
    ``(mlp_delta, residual)`` tuple the hook sums to match HF's ``hook_out`` semantics).
    """
    canonical_name = config["canonical_name"]
    channel = config["channel"]
    materialize = bool(config.get("materialize", False))
    socket = _get_push_socket(channel)

    @torch.no_grad()
    def hook(_module, _inputs, output):
        if not _shared_state["capture_enabled"]:
            return None

        if materialize and isinstance(output, tuple) and len(output) == 2:
            hidden, residual = output
            if not (isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor)):
                return None
            t = hidden + residual
            tuple_tail: tuple = ()
        elif isinstance(output, tuple):
            t = output[0]
            tuple_tail = output[1:]
            if not isinstance(t, torch.Tensor):
                return None
        else:
            t = output
            tuple_tail = ()
            if not isinstance(t, torch.Tensor):
                return None

        spec = _shared_state["interventions"].get(canonical_name)
        modified = _apply_intervention_inline(t, spec) if spec is not None else t

        # Drop on backpressure rather than block the forward.
        try:
            import zmq

            socket.send_pyobj(
                {"name": canonical_name, "tensor": modified.detach().to("cpu").clone()},
                flags=zmq.NOBLOCK,
            )
        except Exception:
            pass

        if spec is None:
            return None
        if materialize:
            # Reconstruct the fused sum so the next layer's input_layernorm sees it.
            _, residual = output  # type: ignore[misc]
            return (modified - residual, residual)
        if isinstance(output, tuple):
            return (modified,) + tuple_tail
        return modified

    return hook


def _get_push_socket(channel: str):
    """Lazily create one PUSH socket per channel, shared across hooks."""
    if channel in _push_sockets:
        return _push_sockets[channel]
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    # Driver binds PULL first; we connect. ZMQ buffers if no peer yet.
    sock.connect(channel)
    _push_sockets[channel] = sock
    return sock


def _apply_intervention_inline(t: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
    """Affine intervention applied in the hook. Validation happens driver-side."""
    op = spec.get("op")
    if op not in SUPPORTED_OPS:
        return t
    if op == "suppress":
        return torch.zeros_like(t)
    if op == "scale":
        return t * float(spec["factor"])
    value = torch.as_tensor(spec["value"], device=t.device, dtype=t.dtype)
    if op == "add":
        return t + value
    if op == "set":
        return torch.zeros_like(t) + value
    return t


# --- Scheduler-attached methods -------------------------------------------------
# :func:`worker_plugin.register` binds these on ``Scheduler`` so the dispatcher's
# ``getattr(self, method)(**parameters)`` reaches them via ``engine.collective_rpc``.
# Return values are discarded (RpcReqOutput is ack-only); tensors flow back on ZMQ.


def tl_set_interventions(self: Any, specs: Dict[str, Dict[str, Any]]) -> None:
    """Replace the active intervention spec set; ``{}`` to clear."""
    _shared_state["interventions"] = dict(specs)


def tl_set_capture_enabled(self: Any, enabled: bool) -> None:
    """Toggle whether hook closures send and apply interventions."""
    _shared_state["capture_enabled"] = bool(enabled)


def tl_clear_state(self: Any) -> None:
    """Reset both interventions and capture-enabled; called at driver teardown."""
    _shared_state["interventions"] = {}
    _shared_state["capture_enabled"] = False


def tl_get_param(self: Any, dotted_name: str, channel: str) -> None:
    """Walk Scheduler→worker→ModelRunner→model for ``dotted_name``; push
    ``{"_param": ..., "tensor": cpu_clone}`` on ``channel``. Silent no-op on miss."""
    model = None
    # Attribute shapes drift release-to-release; try the common ones.
    for tp_attr in ("tp_workers", "tp_worker"):
        tw = getattr(self, tp_attr, None)
        if isinstance(tw, list):
            tw = tw[0] if tw else None
        if tw is None:
            continue
        mr = getattr(tw, "model_runner", None)
        if mr is None:
            continue
        m = getattr(mr, "model", None)
        if m is not None:
            model = m
            break
    if model is None:
        return

    target: Any = model
    for seg in dotted_name.split("."):
        target = target[int(seg)] if seg.isdigit() else getattr(target, seg, None)
        if target is None:
            return

    if not isinstance(target, torch.Tensor):
        return

    sock = _get_push_socket(channel)
    sock.send_pyobj({"_param": dotted_name, "tensor": target.detach().to("cpu").clone()})


# Methods :func:`worker_plugin.register` installs on Scheduler.
SCHEDULER_METHODS = (
    "tl_set_interventions",
    "tl_set_capture_enabled",
    "tl_clear_state",
    "tl_get_param",
)
