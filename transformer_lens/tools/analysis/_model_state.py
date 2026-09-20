"""Shared model-state validation for analysis tools."""

from typing import Any

import torch


def require_eval_mode(model: Any, *, operation: str) -> None:
    """Reject training state anywhere in a wrapped model without mutating it."""
    training_modules: dict[int, str] = {}
    roots = (("", model), ("original_model", getattr(model, "original_model", None)))
    for prefix, root in roots:
        if not isinstance(root, torch.nn.Module):
            continue
        for name, module in root.named_modules():
            if not module.training:
                continue
            qualified_name = ".".join(part for part in (prefix, name) if part)
            training_modules.setdefault(id(module), qualified_name or "<root>")
    if not training_modules:
        return

    names = list(training_modules.values())
    preview = ", ".join(names[:3])
    if len(names) > 3:
        preview += f", and {len(names) - 3} more"
    raise ValueError(
        f"{operation} requires the model and all submodules to be in evaluation "
        f"mode; found training mode at {preview}. Call model.eval() before running "
        "the analysis."
    )
