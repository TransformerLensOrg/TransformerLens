"""Validate intervention specs and key them by ``<layer>:<kind>`` for the provider.

Our HF provider applies interventions as forward-hook affine ops at the residual/
attn/mlp boundaries, so the full vocabulary works. ``resid_mid`` is derived
(capture-only), so intervening on it is rejected.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import hooks

# suppress (→0), scale (factor), add (value), set (value).
SUPPORTED_OPS = frozenset({"suppress", "scale", "add", "set"})


def build_interventions(
    intervene: Mapping[str, Any],
    supported_hook_points: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Validate specs and return ``{wire_key: spec}`` for the provider to apply.

    Rejects callables (remote drivers take specs, not callbacks), bad ops, unknown or
    unsupported hooks, and the capture-only ``resid_mid``.
    """
    out: dict[str, dict[str, Any]] = {}
    for hook_name, spec in intervene.items():
        if callable(spec):
            raise NotImplementedError(
                "InspectDriver requires intervention specs (dict), not callables. "
                "Supported ops: suppress, scale (factor: float), add/set (value: scalar or "
                "width-shaped)."
            )
        if not isinstance(spec, Mapping) or "op" not in spec:
            raise ValueError(
                f"Intervention spec for {hook_name!r} must be a dict with 'op' key; got {spec!r}"
            )
        op = spec["op"]
        if op not in SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported intervention op {op!r} for {hook_name!r}. "
                f"Supported: {sorted(SUPPORTED_OPS)}."
            )
        if hook_name not in supported_hook_points:
            raise ValueError(f"Cannot intervene on {hook_name!r}: not in supported_hook_points.")
        resolved = hooks.resolve(hook_name)
        if resolved is None or resolved[1] not in hooks.INTERVENEABLE_KINDS:
            raise ValueError(
                f"Cannot intervene on {hook_name!r}: capture-only "
                f"(intervene on resid_pre/attn_out/mlp_out/resid_post instead)."
            )
        if op == "scale" and "factor" not in spec:
            raise ValueError(f"Intervention {hook_name!r}: op='scale' requires 'factor' (float).")
        if op in ("add", "set") and "value" not in spec:
            raise ValueError(
                f"Intervention {hook_name!r}: op={op!r} requires 'value' (scalar or width-shaped)."
            )
        pos = spec.get("pos")
        if pos is not None and not (
            isinstance(pos, int)
            or (isinstance(pos, (list, tuple)) and all(isinstance(p, int) for p in pos))
        ):
            raise ValueError(
                f"Intervention {hook_name!r}: 'pos' must be an int or list of ints "
                f"(sequence positions to patch); got {pos!r}."
            )
        layer, kind = resolved
        out[hooks.wire_key(layer, kind)] = dict(spec)
    return out
