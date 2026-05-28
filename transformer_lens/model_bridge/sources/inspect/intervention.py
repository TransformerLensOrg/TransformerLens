"""Validate intervention specs and pack them into the Inspect ``extra_args``.

Our HF provider applies interventions as forward-hook affine ops, so the full
vocabulary is available (unlike vllm-lens, which only adds steering vectors).
The op set mirrors ``vllm.intervention_specs.SUPPORTED_OPS`` — kept local so the
torch-free driver's import chain doesn't drag in the vLLM package.
"""
from __future__ import annotations

from typing import Any, Mapping

# suppress (→0), scale (factor), add (value), set (value). Mirrors the vLLM driver.
SUPPORTED_OPS = frozenset({"suppress", "scale", "add", "set"})


def build_extra_args(
    intervene: Mapping[str, Any],
    supported_hook_points: frozenset[str],
    hook_to_layer: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Validate specs and return ``{layer_index: spec}`` for the provider to apply.

    Rejects callables (remote drivers take specs, not Python callbacks) and
    validates op/params/hook membership — mirrors ``VLLMDriver._validate_interventions``.
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
        if op == "scale" and "factor" not in spec:
            raise ValueError(f"Intervention {hook_name!r}: op='scale' requires 'factor' (float).")
        if op in ("add", "set") and "value" not in spec:
            raise ValueError(
                f"Intervention {hook_name!r}: op={op!r} requires 'value' (scalar or width-shaped)."
            )
        out[str(hook_to_layer[hook_name])] = dict(spec)
    return out
