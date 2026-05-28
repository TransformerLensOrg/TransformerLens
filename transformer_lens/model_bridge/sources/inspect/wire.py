"""Serialization chokepoint for the Inspect activation wire format.

Activations ride in ``ModelOutput.metadata["activations"]`` as
``{"residual_stream": {layer: {"data": <b64>, "dtype": str, "shape": [...]}}}`` —
aligned with the vllm-lens convention so one provider-agnostic driver decodes
both our provider's output and a vllm-lens provider's. Numpy-only (no torch) so
both the torch-using provider and the torch-free driver import it; the single
place to patch on format drift.
"""
from __future__ import annotations

import base64
from typing import Any, Iterable, Mapping

import numpy as np

_RESIDUAL = "residual_stream"


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


def encode_activations(residual_by_layer: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """Per-layer residual streams → the ``metadata["activations"]`` payload."""
    return {_RESIDUAL: {str(layer): encode_array(arr) for layer, arr in residual_by_layer.items()}}


def decode_activations(
    metadata: Mapping[str, Any] | None, layers: Iterable[int]
) -> dict[int, np.ndarray]:
    """Pull the requested layers' residual streams out of ``metadata["activations"]``.

    Tolerates int or str layer keys (our provider emits str; a peer provider may
    differ). Missing layers are silently skipped — the driver decides what to do.
    """
    residual = ((metadata or {}).get("activations") or {}).get(_RESIDUAL, {})
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        entry = residual.get(str(layer), residual.get(layer))
        if entry is not None:
            out[layer] = decode_array(entry)
    return out
