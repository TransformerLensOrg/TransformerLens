"""Serialization chokepoint for the Inspect activation wire format.

Activations ride in ``ModelOutput.metadata["activations"]`` as a flat
``{"<layer>:<kind>": {"data": <b64>, "dtype": str, "shape": [...]}}`` map (keys
are :func:`hooks.wire_key`). For vllm-lens interop, decode also understands its
*documented* nested ``{"residual_stream": {layer: ...}}`` shape (mapped to
``resid_post``) — unverified against a live vllm-lens provider.
Numpy-only (no torch) so both the torch-using provider and the torch-free driver
import it; the single place to patch on format drift.
"""
from __future__ import annotations

import base64
from typing import Any, Iterable, Mapping

import numpy as np

_RESIDUAL = "residual_stream"  # vllm-lens's nested key (interop decode only)


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    """numpy array → ``{"data": b64, "dtype": str, "shape": [...]}``."""
    contiguous = np.ascontiguousarray(arr)
    return {
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
    }


def decode_array(entry: Any) -> np.ndarray:
    """Inverse of :func:`encode_array`; passes an already-decoded ndarray through."""
    if isinstance(entry, np.ndarray):
        return entry
    raw = base64.b64decode(entry["data"])
    # frombuffer is read-only; copy so the downstream torch tensor is writable.
    return np.frombuffer(raw, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"]).copy()


def encode_activations(captured: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """``{wire_key: array}`` → the ``metadata["activations"]`` payload."""
    return {key: encode_array(arr) for key, arr in captured.items()}


def decode_activations(
    metadata: Mapping[str, Any] | None, wire_keys: Iterable[str]
) -> dict[str, np.ndarray]:
    """Pull the requested ``<layer>:<kind>`` keys out of ``metadata["activations"]``,
    falling back to the nested ``residual_stream`` for ``resid_post``. Missing keys are
    skipped — the caller decides."""
    activations = (metadata or {}).get("activations") or {}
    residual = activations.get(_RESIDUAL, {})
    out: dict[str, np.ndarray] = {}
    for key in wire_keys:
        if key in activations:
            out[key] = decode_array(activations[key])
            continue
        layer, _, kind = key.partition(":")
        if kind == "resid_post":
            entry = residual.get(layer, residual.get(_safe_int(layer)))
            if entry is not None:
                out[key] = decode_array(entry)
    return out


def _safe_int(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        return value
