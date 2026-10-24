"""Shared model-state and argument validation for analysis tools."""

from numbers import Integral
from typing import Any, List, Optional

import torch

# Block variants with no attn_out + mlp_out split and no resid_mid hook.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")
_ID_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)


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


def hybrid_block_types(model: Any) -> List[str]:
    """Sorted distinct ``model.layer_types()`` entries naming an SSM or linear-attention block."""
    layer_types = model.layer_types() if hasattr(model, "layer_types") else []
    return sorted(
        {t for t in layer_types if any(part in _HYBRID_VARIANT_NAMES for part in t.split("+"))}
    )


def compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Keep float64; run every other dtype in float32, which all backends' linalg support."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def require_int(
    value: Any,
    name: str,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
    message: Optional[str] = None,
) -> int:
    """Return ``value`` as an int in ``[minimum, maximum]``; bools are refused as integers."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(message or f"{name} must be an integer, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is not None:
            expected = f"between {minimum} and {maximum}"
        elif minimum == 1:
            expected = "a positive integer"
        elif minimum == 0:
            expected = "a nonnegative integer"
        else:
            expected = f"an integer at least {minimum}"
        raise ValueError(message or f"{name} must be {expected}, got {value!r}")
    return int(value)


# Typed Any so the checks below, not a runtime type-hint checker, name a bad token.
def to_token_ids(model: Any, tokens: Any) -> List[int]:
    """Resolve one token, or a sequence of them, into in-vocabulary token ids.

    A token is a single-token string or an integer id: a Python or numpy integer, or a 0-d
    integer tensor. A bare token and a one-element sequence holding it resolve alike.
    """
    if isinstance(tokens, (str, Integral)) or (
        isinstance(tokens, torch.Tensor) and tokens.ndim == 0
    ):
        tokens = [tokens]
    ids: List[int] = []
    for token in tokens:
        if isinstance(token, torch.Tensor) and token.ndim == 0 and token.dtype in _ID_DTYPES:
            token = token.item()
        if isinstance(token, str):
            ids.append(int(model.to_single_token(token)))
        elif isinstance(token, bool):
            raise ValueError(f"token {token!r} must be a string or integer token id, not bool")
        elif isinstance(token, Integral):
            ids.append(int(token))
        else:
            # int() would silently truncate a float id.
            raise TypeError(
                f"tokens must be strings or integer token ids, got {type(token).__name__}"
            )
    if not ids:
        raise ValueError("tokens must contain at least one token")
    d_vocab = model.W_U.shape[1]
    invalid = [token_id for token_id in ids if not 0 <= token_id < d_vocab]
    if invalid:
        raise ValueError(f"token ids {invalid} out of range for vocabulary size {d_vocab}")
    return ids
