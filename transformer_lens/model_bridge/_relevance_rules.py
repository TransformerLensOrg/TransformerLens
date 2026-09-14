"""Forward-equivalent relevance-rule primitives, plus the scoped context that installs them.

Each primitive is a ``torch.autograd.Function`` that reproduces its native forward
value exactly while replacing the backward pass with the rule's closed-form VJP.
``use_relevance_rules`` installs these rules on a model's canonical mount points only
for the duration of a ``with`` block, targeting components positionally (by mount
name, never by class) and reporting which mounts were installed versus skipped.
"""

import dataclasses
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch
import torch.nn as nn


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


@dataclasses.dataclass(frozen=True)
class RelevanceRules:
    """Which relevance rules to request for the duration of a ``use_relevance_rules`` scope.

    Each field names a rule kind. Setting it ``True`` requests that rule wherever a
    component at that kind's canonical mount point implements ``_RelevanceRuleCapable``.
    Unset fields (the default) leave the corresponding components untouched.
    """

    normalization: bool = False
    activation: bool = False
    multiplicative_gate: bool = False
    attention: bool = False


@dataclasses.dataclass
class RelevanceRuleCoverage:
    """Which canonical mounts a ``use_relevance_rules`` scope installed versus skipped.

    ``installed`` holds the dotted path of every mount where a requested rule kind was
    actually enabled. ``skipped`` holds the dotted path of every mount that matched a
    requested kind's canonical mount name but did not implement the relevance-rule
    protocol there, so no rule could be installed.
    """

    installed: Tuple[str, ...]
    skipped: Tuple[str, ...]


@runtime_checkable
class _RelevanceRuleCapable(Protocol):
    """Structural contract a component must satisfy to accept a relevance rule.

    ``_relevance_rule_kind`` names which ``RelevanceRules`` field the component answers
    to; ``_enable_relevance_rule``/``_disable_relevance_rule`` toggle the rule without
    touching model configuration, so the component's own state is the only thing that
    changes and only for the scope's duration.
    """

    _relevance_rule_kind: str

    def _enable_relevance_rule(self) -> None:
        ...

    def _disable_relevance_rule(self) -> None:
        ...


# Canonical mount name per rule kind. Targeting is positional: a component is only
# considered for a kind when it sits at that kind's mount name, never by isinstance,
# so a same-class component mounted elsewhere (for example a q_norm sharing
# NormalizationBridge's class) is left untouched.
_CANONICAL_MOUNTS: Mapping[str, Tuple[str, ...]] = {
    "normalization": ("ln1", "ln2"),
}


def _acquire_rule(module: _RelevanceRuleCapable) -> None:
    """Enable ``module``'s rule only on the outermost scope that requests it."""
    count = getattr(module, "_relevance_rule_refcount", 0)
    if count == 0:
        module._enable_relevance_rule()
    setattr(module, "_relevance_rule_refcount", count + 1)


def _release_rule(module: _RelevanceRuleCapable) -> None:
    """Disable ``module``'s rule only once the innermost scope that requested it exits."""
    count = getattr(module, "_relevance_rule_refcount", 0) - 1
    setattr(module, "_relevance_rule_refcount", max(count, 0))
    if count <= 0:
        module._disable_relevance_rule()


@contextmanager
def use_relevance_rules(model: nn.Module, rules: RelevanceRules) -> Iterator[RelevanceRuleCoverage]:
    """Install the requested relevance rules on ``model`` only for this scope.

    Targeting is positional: a component is considered for a rule kind only when it
    sits at that kind's canonical mount name (never by class). A component at a
    canonical mount that does not implement ``_RelevanceRuleCapable`` for the
    requested kind is reported as skipped rather than installed or raising. Scopes
    over the same model are reference-counted, so an inner scope's exit never
    disables a rule an outer scope still needs. No model configuration is mutated;
    the only state that changes lives on the participating components, and only for
    the scope's duration.
    """
    requested_kinds = [
        field.name for field in dataclasses.fields(rules) if getattr(rules, field.name)
    ]

    installed: List[Tuple[str, _RelevanceRuleCapable]] = []
    skipped: List[str] = []
    for kind in requested_kinds:
        mount_names = _CANONICAL_MOUNTS.get(kind, ())
        for name, module in model.named_modules():
            if name.rsplit(".", 1)[-1] not in mount_names:
                continue
            if isinstance(module, _RelevanceRuleCapable) and module._relevance_rule_kind == kind:
                installed.append((name, module))
            else:
                skipped.append(name)

    for _, module in installed:
        _acquire_rule(module)
    try:
        yield RelevanceRuleCoverage(
            installed=tuple(name for name, _ in installed),
            skipped=tuple(skipped),
        )
    finally:
        for _, module in installed:
            _release_rule(module)
