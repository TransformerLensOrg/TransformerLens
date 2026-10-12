"""Relevance-rule autograd primitives (native forward, rule VJP backward) and the scoped
``use_relevance_rules``, which reports installed versus skipped canonical mounts."""

import dataclasses
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch
import torch.nn as nn


def ln_rule_grad(grad_output: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """LN-rule VJP: divide by ``denom`` without differentiating through it; also used by
    ``NormalizationBridge``, which wraps the norm's own forward instead of ``_LNRule``."""
    return grad_output / denom


class _LNRule(torch.autograd.Function):
    """LN-rule: forward is ``numerator / denom``; the VJP treats ``denom`` as constant."""

    @staticmethod
    def forward(ctx: Any, numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(denom)
        return numerator / denom

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (denom,) = ctx.saved_tensors
        return ln_rule_grad(grad_output, denom), None


def ln_rule(numerator: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """Apply the LN-rule: native division forward, denom-as-constant backward."""
    result: torch.Tensor = _LNRule.apply(numerator, denom)
    return result


class _IdentityRule(torch.autograd.Function):
    """Identity-rule: forward is the native activation; the VJP is ``grad_out * f(x) / x``,
    using the ``x == 0`` limit ``0.5`` (exact for SiLU and both GELU variants)."""

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


class _ScaleGradient(torch.autograd.Function):
    """Identity forward; the VJP scales the incoming gradient by a constant factor."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, factor: float) -> torch.Tensor:
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return ctx.factor * grad_output, None


def scale_gradient(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Identity forward whose VJP scales by ``factor``; 0.5 on an opaque ``u * v`` output
    reproduces the Half-rule, since halving the product's gradient halves both terms."""
    result: torch.Tensor = _ScaleGradient.apply(x, factor)
    return result


class RelevanceRuleConflictError(RuntimeError):
    """A hook would silently break a rule-active forward/backward invariant."""


class RelevanceRuleUnsupportedError(RuntimeError):
    """A component lists a requested kind in ``_relevance_rule_unsupported_kinds``; raised at
    scope entry, unlike kinds merely absent from ``_relevance_rule_kinds`` (reported skipped)."""


@dataclasses.dataclass(frozen=True)
class RelevanceRules:
    """Rule kinds to request for a ``use_relevance_rules`` scope; unset kinds stay untouched."""

    normalization: bool = False
    activation: bool = False
    multiplicative_gate: bool = False
    attention: bool = False


@dataclasses.dataclass(frozen=True)
class _RelevanceRuleCoverageEntry:
    """A rule kind and its canonical component path."""

    kind: str
    path: str


@dataclasses.dataclass(frozen=True)
class RelevanceRuleCoverage:
    """Requested rule kinds installed or skipped at canonical component paths."""

    installed: Tuple[_RelevanceRuleCoverageEntry, ...]
    skipped: Tuple[_RelevanceRuleCoverageEntry, ...]


@runtime_checkable
class _RelevanceRuleCapable(Protocol):
    """Rule-capable component; enable/disable toggle only the given kind's state. Optional
    ``_relevance_rule_unsupported_kinds`` stays off the protocol so isinstance still holds."""

    _relevance_rule_kinds: Tuple[str, ...]

    def _enable_relevance_rule(self, kind: str) -> None:
        ...

    def _disable_relevance_rule(self, kind: str) -> None:
        ...


# Positional targeting (never isinstance) leaves a same-class q_norm untouched. Sandwich
# post-norms (ln1_post/ln2_post) are included to match the pinned RelP reference.
_CANONICAL_MOUNTS: Mapping[str, Tuple[str, ...]] = {
    "normalization": ("ln1", "ln2", "ln1_post", "ln2_post"),
    "activation": ("mlp",),
    "multiplicative_gate": ("mlp",),
}


def _iter_canonical_mount_candidates(
    model: nn.Module, mount_names: Tuple[str, ...]
) -> Iterator[Tuple[str, _RelevanceRuleCapable]]:
    """Yield each distinct module at one of ``mount_names`` once, under its shallowest path."""
    best_by_id: Dict[int, Tuple[str, _RelevanceRuleCapable]] = {}
    # Default dedup keeps the raw HF alias (_original_component.input_layernorm) visited
    # first, hiding the canonical name; the shallowest path also beats HF's own "mlp".
    for name, module in model.named_modules(remove_duplicate=False):
        if name.rsplit(".", 1)[-1] not in mount_names:
            continue
        existing = best_by_id.get(id(module))
        if existing is None or name.count(".") < existing[0].count("."):
            best_by_id[id(module)] = (name, module)
    yield from best_by_id.values()


def _acquire_rule(module: _RelevanceRuleCapable, kind: str) -> None:
    """Enable ``kind`` on the outermost scope; refcounted per kind, not per module, so one
    kind's inner exit never disables another kind nested on the same gated-MLP node."""
    counts: Dict[str, int] = getattr(module, "_relevance_rule_refcounts", None) or {}
    count = counts.get(kind, 0)
    if count == 0:
        module._enable_relevance_rule(kind)
    counts[kind] = count + 1
    setattr(module, "_relevance_rule_refcounts", counts)


def _release_rule(module: _RelevanceRuleCapable, kind: str) -> None:
    """Disable ``module``'s ``kind`` rule only once its innermost scope exits."""
    counts: Dict[str, int] = getattr(module, "_relevance_rule_refcounts", None) or {}
    count = counts.get(kind, 0) - 1
    counts[kind] = max(count, 0)
    setattr(module, "_relevance_rule_refcounts", counts)
    if count <= 0:
        module._disable_relevance_rule(kind)


@contextmanager
def use_relevance_rules(model: nn.Module, rules: RelevanceRules) -> Iterator[RelevanceRuleCoverage]:
    """Install the requested rules for this scope only and yield installed/skipped coverage;
    nested scopes are refcounted and only component state (never model config) changes."""
    requested_kinds = [
        field.name for field in dataclasses.fields(rules) if getattr(rules, field.name)
    ]

    unmapped_kinds = [kind for kind in requested_kinds if kind not in _CANONICAL_MOUNTS]
    if unmapped_kinds:
        joined = ", ".join(repr(kind) for kind in unmapped_kinds)
        raise ValueError(
            f"No canonical mount is defined for relevance-rule kind(s) {joined}. "
            "Such a kind has no mount to target, so it would install nothing and "
            "silently return an empty coverage report instead of applying the rule."
        )

    installed: List[Tuple[str, _RelevanceRuleCapable, str]] = []
    skipped: List[_RelevanceRuleCoverageEntry] = []
    for kind in requested_kinds:
        for name, module in _iter_canonical_mount_candidates(model, _CANONICAL_MOUNTS[kind]):
            if isinstance(module, _RelevanceRuleCapable) and kind in module._relevance_rule_kinds:
                installed.append((name, module, kind))
                continue
            unsupported_kinds = getattr(module, "_relevance_rule_unsupported_kinds", ())
            if isinstance(module, _RelevanceRuleCapable) and kind in unsupported_kinds:
                raise RelevanceRuleUnsupportedError(
                    f"{name!r} ({type(module).__name__}) cannot install the {kind!r} "
                    "relevance rule: unsupported configuration for this component."
                )
            skipped.append(_RelevanceRuleCoverageEntry(kind=kind, path=name))

    for _, module, kind in installed:
        _acquire_rule(module, kind)
    try:
        yield RelevanceRuleCoverage(
            installed=tuple(
                _RelevanceRuleCoverageEntry(kind=kind, path=name) for name, _, kind in installed
            ),
            skipped=tuple(skipped),
        )
    finally:
        for _, module, kind in installed:
            _release_rule(module, kind)
