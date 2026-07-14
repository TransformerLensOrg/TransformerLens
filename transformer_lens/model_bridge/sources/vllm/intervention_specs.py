"""Intervention op vocabulary + spec validation shared by every producer path
(VLLMDriver, the worker RPCs — which also cover the Inspect vLLM provider).

Adding a new op requires (a) listing it here, (b) allowing its keys in
``validate_spec``, (c) handling it in ``worker_extension._apply_intervention``.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

SUPPORTED_OPS = frozenset({"suppress", "scale", "add", "set"})


def validate_spec(hook_name: str, spec: Any, *, width: int | None = None) -> dict:
    """Validate one declarative intervention spec; returns a plain-dict copy.

    ``width`` enables the value-shape check where the caller knows the hook width.
    """
    if callable(spec):
        raise NotImplementedError(
            "vLLM accepts intervention specs (dict), not callables. "
            "Supported ops: suppress, scale (factor: float), add (value: scalar or "
            "width-shaped), set (value: scalar or width-shaped)."
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
    if op == "scale" and "factor" not in spec:
        raise ValueError(f"Intervention {hook_name!r}: op='scale' requires 'factor' (float).")
    if op in ("add", "set") and "value" not in spec:
        raise ValueError(
            f"Intervention {hook_name!r}: op={op!r} requires 'value' "
            "(scalar or width-shaped tensor/list)."
        )
    # Unknown keys must fail loud: a typo'd "position"/"positions" would otherwise
    # silently become a whole-sequence edit.
    allowed = {"op", "pos"}
    if op == "scale":
        allowed.add("factor")
    if op in ("add", "set"):
        allowed.add("value")
    extra = set(spec) - allowed
    if extra:
        raise ValueError(
            f"Intervention {hook_name!r}: unknown spec key(s) {sorted(extra)}; "
            f"allowed for op={op!r}: {sorted(allowed)}."
        )
    value = spec.get("value")
    if value is not None and not isinstance(value, (int, float)):
        try:
            n_elements = int(torch.as_tensor(value).numel())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Intervention {hook_name!r}: 'value' must be a scalar or a "
                f"width-shaped tensor/list; got {type(value).__name__}."
            ) from exc
        if width is not None and n_elements != width:
            # A mis-shaped value would otherwise surface as a broadcast error
            # mid-forward — or broadcast along the wrong axis in square cases.
            raise ValueError(
                f"Intervention {hook_name!r}: 'value' has {n_elements} elements "
                f"but the hook width is {width}."
            )
    pos = spec.get("pos")
    if pos is not None:
        valid_pos = isinstance(pos, int) and not isinstance(pos, bool)
        if not valid_pos and isinstance(pos, (list, tuple)):
            valid_pos = all(isinstance(p, int) and not isinstance(p, bool) for p in pos)
        if not valid_pos:
            raise ValueError(
                f"Intervention {hook_name!r}: 'pos' must be an int or list of ints "
                f"(sequence positions to patch); got {pos!r}."
            )
        negative = [p for p in ([pos] if isinstance(pos, int) else pos) if p < 0]
        if negative:
            raise ValueError(
                f"Intervention {hook_name!r}: 'pos' must be non-negative; got {negative}."
            )
    return dict(spec)
