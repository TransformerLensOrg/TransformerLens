"""Chokepoint for ``RpcReqInput`` cross-process method calls. The Engine-side
send path (``tokenizer_manager.send_rpc_request`` vs a public ``Engine.rpc``
helper) is the v1 driver's biggest uncertainty — :func:`call` is where it
resolves on live verification."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def call(
    engine: Any,
    method: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """Dispatch ``method`` on the worker via SGLang's RPC; return its result.
    Prefers a vLLM-style ``engine.collective_rpc`` if SGLang grows one, else
    falls back to the documented ``RpcReqInput`` + ``tokenizer_manager`` path."""
    if hasattr(engine, "collective_rpc"):
        return engine.collective_rpc(method, args=tuple(parameters.values()) if parameters else ())

    from .internals import rpc_classes

    RpcReqInput, _ = rpc_classes()
    req = RpcReqInput(method=method, parameters=parameters or {})

    tm = getattr(engine, "tokenizer_manager", None)
    if tm is None:
        raise RuntimeError("Engine has no tokenizer_manager; RPC dispatch unavailable.")
    # Send-method name has moved release-to-release; re-validate on version bumps.
    for send_method in ("send_rpc_request", "rpc_request", "send_rpc"):
        send = getattr(tm, send_method, None)
        if send is not None:
            return send(req)
    raise RuntimeError("Could not locate the RPC send method on tokenizer_manager.")


def call_with_prompt_lens(
    engine: Any,
    method: str,
    prompt_lens: List[int],
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for the ``tl_read_captures(prompt_lens, names)`` call."""
    return call(engine, method, {"prompt_lens": prompt_lens, "names": names})


def reset_capture_flags(engine: Any) -> None:
    """Open every per-hook capture gate before a capture-needing forward."""
    call(engine, "tl_reset_capture_flags", parameters=None)


def set_interventions(engine: Any, specs: Dict[str, Dict[str, Any]]) -> None:
    """Push the full intervention spec set (or ``{}`` to reset to identity)."""
    call(engine, "tl_set_interventions", {"specs": specs})


def reset_counter(engine: Any) -> None:
    """Zero the hook-fire counter before a capture forward (fires-once check)."""
    call(engine, "tl_reset_counter", parameters=None)


def read_counter(engine: Any) -> int:
    """Read the hook-fire counter (fires-once verification)."""
    result = call(engine, "tl_read_counter", parameters=None)
    return int(result) if result is not None else 0


def remove_hooks(engine: Any) -> None:
    """Detach all capture hooks and drop buffer references on the worker. Idempotent."""
    call(engine, "tl_remove_hooks", parameters=None)
