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
    """Core LN-rule VJP: divide by ``denom`` without differentiating through it.

    Shared by the ``ln_rule`` primitive below and by any integration (such as
    ``NormalizationBridge``) that wraps a component's own native forward call
    instead of reproducing the division itself.
    """
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
    """Pass ``x`` through unchanged while scaling its gradient by ``factor``.

    The Half-rule on a product ``u * v`` halves each ordinary product-rule term,
    which is the same as halving the single gradient that enters the product before
    it splits. When the product is computed inside an opaque module the bridge cannot
    reach term by term (its native forward is called as one unit), scaling the
    gradient entering the product by ``0.5`` reproduces the Half-rule at that point
    without altering the native forward value.
    """
    result: torch.Tensor = _ScaleGradient.apply(x, factor)
    return result


class RelevanceRuleConflictError(RuntimeError):
    """A hook would silently break a rule-active forward/backward invariant.

    Raised instead of the ordinary warn-and-fall-back a component would use when
    no rule is active, since falling back while a rule is active would compose the
    rule with the hook edit and break the bit-identical-forward guarantee.
    """


class RelevanceRuleUnsupportedError(RuntimeError):
    """A requested relevance rule cannot be installed on an otherwise-capable component.

    Raised at ``use_relevance_rules`` entry, before any forward or backward pass, when
    a component reports the requested kind in its own ``_relevance_rule_unsupported_kinds``
    -- for example a gated-MLP recompute path backed by an unrecognized weight-orientation
    class, or an activation form the Identity-rule does not support. Distinct from a
    kind that is simply absent from ``_relevance_rule_kinds`` without being named there
    (reported ``skipped``, not raised): that covers a component not implementing the
    protocol at all, or one whose mount genuinely never deals with the kind (for example
    normalization on a dispatch path the LN-rule does not wrap), both benign
    non-applicability rather than a rule request the component was expected to honor.
    """


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

    ``_relevance_rule_kinds`` names every ``RelevanceRules`` field the component
    answers to at its current mount -- a gated-MLP node answers to both
    "activation" (Identity-rule on its activation function) and
    "multiplicative_gate" (Half-rule on its gate*up product) independently, since
    either can be requested without the other. ``_enable_relevance_rule``/
    ``_disable_relevance_rule`` take the specific kind being toggled and touch
    only that kind's state, without touching model configuration, so the
    component's own state is the only thing that changes and only for the
    scope's duration.

    A component may optionally also define ``_relevance_rule_unsupported_kinds``
    (a ``Tuple[str, ...]``, not part of this structural protocol so components that
    omit it stay isinstance-compatible) naming kinds it is expected to honor at its
    mount but currently cannot -- ``use_relevance_rules`` raises
    ``RelevanceRuleUnsupportedError`` for those instead of reporting them skipped.
    """

    _relevance_rule_kinds: Tuple[str, ...]

    def _enable_relevance_rule(self, kind: str) -> None:
        ...

    def _disable_relevance_rule(self, kind: str) -> None:
        ...


# Canonical mount name per rule kind. Targeting is positional: a component is only
# considered for a kind when it sits at that kind's mount name, never by isinstance,
# so a same-class component mounted elsewhere (for example a q_norm sharing
# NormalizationBridge's class) is left untouched.
_CANONICAL_MOUNTS: Mapping[str, Tuple[str, ...]] = {
    "normalization": ("ln1", "ln2"),
    "activation": ("mlp",),
    "multiplicative_gate": ("mlp",),
}


def _iter_canonical_mount_candidates(
    model: nn.Module, mount_names: Tuple[str, ...]
) -> Iterator[Tuple[str, _RelevanceRuleCapable]]:
    """Yield each distinct module reachable at one of ``mount_names``, once.

    A bridge component reachable at a canonical mount name (for example
    ``blocks.0.ln1``) is also reachable, under the same parent, through the
    raw HF module tree the bridge wraps in place (for example
    ``blocks.0._original_component.input_layernorm``) -- both names resolve to
    the identical object. ``nn.Module.named_modules()`` deduplicates by object
    identity and keeps only whichever path it visits first, which is the raw
    HF-attribute path (registered before the canonical alias), so on a real
    assembled model the canonical name is silently never seen. Walking with
    ``remove_duplicate=False`` restores every path so the canonical name is
    visible, and picking the fewest-dot-separated-segments path per object
    (breaking a tie between two paths that both happen to end in a mount name,
    such as ``mlp``, which HF's own attribute name also frequently matches)
    reports the shallower, canonical-looking path rather than an internal one.
    """
    best_by_id: Dict[int, Tuple[str, _RelevanceRuleCapable]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if name.rsplit(".", 1)[-1] not in mount_names:
            continue
        existing = best_by_id.get(id(module))
        if existing is None or name.count(".") < existing[0].count("."):
            best_by_id[id(module)] = (name, module)
    yield from best_by_id.values()


def _acquire_rule(module: _RelevanceRuleCapable, kind: str) -> None:
    """Enable ``module``'s ``kind`` rule only on the outermost scope that requests it.

    Refcounted per kind, not per module: a gated-MLP node can have its
    "activation" rule and "multiplicative_gate" rule independently nested to
    different depths, so one kind's inner exit must never disable the other.
    """
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
    """Install the requested relevance rules on ``model`` only for this scope.

    Targeting is positional: a component is considered for a rule kind only when it
    sits at that kind's canonical mount name (never by class). A canonical mount
    occupied by a component that does not implement ``_RelevanceRuleCapable``, or
    whose ``_relevance_rule_kinds`` simply omits the requested kind, is reported as
    skipped -- both are benign non-applicability, covering a structurally different
    architecture or a mount whose current dispatch path the rule does not wrap. A
    component that additionally names the requested kind in its own
    ``_relevance_rule_unsupported_kinds`` raises ``RelevanceRuleUnsupportedError``
    instead: that names a kind the component is expected to honor at this mount but
    cannot given its current configuration, so silently skipping it would let
    analysis proceed as if the caller had never asked. Scopes over the same model
    are reference-counted, so an inner scope's exit never disables a rule an outer
    scope still needs. No model configuration is mutated; the only state that
    changes lives on the participating components, and only for the scope's
    duration.
    """
    requested_kinds = [
        field.name for field in dataclasses.fields(rules) if getattr(rules, field.name)
    ]

    installed: List[Tuple[str, _RelevanceRuleCapable, str]] = []
    skipped: List[str] = []
    for kind in requested_kinds:
        mount_names = _CANONICAL_MOUNTS.get(kind, ())
        for name, module in _iter_canonical_mount_candidates(model, mount_names):
            if isinstance(module, _RelevanceRuleCapable) and kind in module._relevance_rule_kinds:
                installed.append((name, module, kind))
                continue
            unsupported_kinds = getattr(module, "_relevance_rule_unsupported_kinds", ())
            if isinstance(module, _RelevanceRuleCapable) and kind in unsupported_kinds:
                raise RelevanceRuleUnsupportedError(
                    f"{name!r} ({type(module).__name__}) cannot install the {kind!r} "
                    "relevance rule: unsupported configuration for this component."
                )
            skipped.append(name)

    for _, module, kind in installed:
        _acquire_rule(module, kind)
    try:
        yield RelevanceRuleCoverage(
            installed=tuple(name for name, _, _ in installed),
            skipped=tuple(skipped),
        )
    finally:
        for _, module, kind in installed:
            _release_rule(module, kind)
