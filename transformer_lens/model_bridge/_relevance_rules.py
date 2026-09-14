"""Forward-equivalent relevance-rule primitives for custom-VJP relevance propagation.

Each primitive is a ``torch.autograd.Function`` that reproduces its native forward
value exactly while replacing the backward pass with the rule's closed-form VJP.
"""

from typing import Any, Callable, Tuple

import torch


class _LNRule(torch.autograd.Function):
    """LN-rule: forward is ``numerator / denom``; the VJP treats ``denom`` as constant."""

    @staticmethod
    def forward(ctx: Any, numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(denom)
        return numerator / denom

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (denom,) = ctx.saved_tensors
        return grad_output / denom, None


def ln_rule(numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Apply the LN-rule: native division forward, denom-as-constant backward."""
    result: torch.Tensor = _LNRule.apply(numerator, denom)
    return result


class _IdentityRule(torch.autograd.Function):
    """Identity-rule: forward is the native activation; the VJP is ``grad_out * phi(x)``.

    ``phi`` is ``f(x) / x``, the removable singularity at ``x == 0`` filled in with its
    limit ``0.5``. This holds for any elementwise activation with ``f(0) == 0`` and a
    well-defined derivative at zero, which covers SiLU and both GELU variants.
    """

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, act_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        y = act_fn(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x, y = ctx.saved_tensors
        safe_x = torch.where(x == 0, torch.ones_like(x), x)
        phi = torch.where(x == 0, torch.full_like(x, 0.5), y / safe_x)
        return grad_output * phi, None


def identity_rule(x: torch.Tensor, act_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply the Identity-rule for an elementwise activation with ``f(0) == 0``."""
    result: torch.Tensor = _IdentityRule.apply(x, act_fn)
    return result


class _HalfRule(torch.autograd.Function):
    """Half-rule: forward is ``u * v``; the VJP halves each ordinary product-rule term."""

    @staticmethod
    def forward(ctx: Any, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(u, v)
        return u * v

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u, v = ctx.saved_tensors
        return 0.5 * grad_output * v, 0.5 * grad_output * u


def half_rule(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply the Half-rule: native product forward, evenly split backward."""
    result: torch.Tensor = _HalfRule.apply(u, v)
    return result
