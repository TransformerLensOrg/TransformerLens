"""Restore-guaranteed in-place parameter swapping.

``torch.func.functional_call`` (and ``torch.nn.utils.stateless`` generally)
fails to restore parameters on module trees that register the same submodule
under more than one name: the tied-weight machinery swaps the single underlying
slot once per alias, so the second swap stashes the override as the "original"
and restoration installs the override permanently. ``TransformerBridge`` trees
have exactly that shape — every replaced component is registered both in the
wrapped HF tree and as a bridge submodule — so stateless reparametrization
through a bridge silently corrupts it. This module provides the supported
alternative: an in-place value swap whose restore is guaranteed by construction.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def temporarily_swap_parameter(parameter: Any, new_value: Any) -> Iterator[torch.nn.Parameter]:
    """Swap a parameter's value in place and restore it on exit, even on error.

    The parameter object is never replaced, so aliased registrations, optimizer
    references, hooks, and ``requires_grad`` state all stay intact. ``.grad`` is
    untouched. The restore runs in a ``finally`` block.

    Args:
        parameter: The live ``torch.nn.Parameter`` to modify.
        new_value: Replacement values with the same shape and dtype. It may live
            on a different device; values are copied in.

    Yields:
        The same parameter, holding ``new_value`` for the duration of the block.

    Raises:
        TypeError: If ``parameter`` is not a ``torch.nn.Parameter`` or
            ``new_value`` is not a ``torch.Tensor``.
        ValueError: If shapes or dtypes differ — silent casts would make the
            swapped forward incomparable to the caller's intent.
    """
    # Any-typed params keep these manual checks live under the project's
    # beartype instrumentation; the docstring states the real types.
    if not isinstance(parameter, torch.nn.Parameter):
        raise TypeError(f"parameter must be a torch.nn.Parameter; got {type(parameter).__name__}")
    if isinstance(new_value, torch.nn.Parameter) or not isinstance(new_value, torch.Tensor):
        raise TypeError(f"new_value must be a plain torch.Tensor; got {type(new_value).__name__}")
    if new_value.shape != parameter.shape:
        raise ValueError(
            f"new_value shape {tuple(new_value.shape)} must match parameter shape "
            f"{tuple(parameter.shape)}"
        )
    if new_value.dtype != parameter.dtype:
        raise ValueError(
            f"new_value dtype {new_value.dtype} must match parameter dtype {parameter.dtype}"
        )
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.copy_(new_value)
        yield parameter
    finally:
        with torch.no_grad():
            parameter.copy_(original)
