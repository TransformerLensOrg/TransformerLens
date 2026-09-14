"""Tests for the scoped relevance-rule context: forward-identity, cleanup, and nesting.

The fixture component below installs the real LN-rule primitive
(``transformer_lens.model_bridge._relevance_rules.ln_rule``) through the protocol
``use_relevance_rules`` relies on to find and toggle rule-capable components. That keeps
these tests focused on the scoping mechanics -- forward identity, gradient restoration,
nested contexts, exception safety, and positional (not class-based) targeting -- rather
than on any concrete NormalizationBridge or gated-MLP integration, which land in later
commits.

The scoped context does not exist yet, so this fails collection with a single
ImportError -- the expected red state before the context is implemented.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleCoverage,
    RelevanceRules,
    ln_rule,
    use_relevance_rules,
)


class _FakeNormComponent(nn.Module):
    """A minimal ln1/ln2-style target: installs the real LN-rule primitive on request."""

    _relevance_rule_kind = "normalization"

    def __init__(self, eps: float = 1e-2):
        super().__init__()
        # Composing this division with a plain second normalization and a linear
        # readout makes the LN-rule's denom-as-constant correction shrink linearly
        # with eps, so a realistic normalization epsilon (1e-6) would leave the
        # correction too small for torch.allclose to detect reliably end to end.
        self.eps = eps
        self._rule_active = False

    def _enable_relevance_rule(self) -> None:
        self._rule_active = True

    def _disable_relevance_rule(self) -> None:
        self._rule_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denom = x.abs().mean(dim=-1, keepdim=True) + self.eps
        if self._rule_active:
            return ln_rule(x, denom)
        return x / denom


class _PlainMount(nn.Module):
    """Occupies a targeted mount name but implements no relevance-rule protocol."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyBlock(nn.Module):
    """Mimics a block's ln1/ln2 mount points plus a same-class q_norm and an MLP.

    ``q_norm`` uses the identical fake-normalization class as ``ln1``/``ln2`` so tests
    can assert that targeting is positional (by mount name) rather than class-based.
    """

    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = _FakeNormComponent()
        self.q_norm = _FakeNormComponent()
        self.ln2: nn.Module = _FakeNormComponent()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.q_norm(x)
        x = self.ln2(x)
        return self.mlp(x)


def _tiny_block() -> _TinyBlock:
    block = _TinyBlock()
    with torch.no_grad():
        block.mlp.weight.copy_(torch.eye(4) * 0.5)
        block.mlp.bias.zero_()
    return block


class _Boom(Exception):
    """Marker exception raised inside a context to test cleanup on failure."""


def test_relevance_rules_defaults_to_no_rules():
    rules = RelevanceRules()
    assert rules.normalization is False
    assert rules.activation is False
    assert rules.multiplicative_gate is False
    assert rules.attention is False


def test_relevance_rules_is_frozen():
    rules = RelevanceRules(normalization=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules.normalization = False  # type: ignore[misc]


def test_forward_is_identical_while_rule_active():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)
    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        active = block(x)
    assert torch.equal(active, baseline)


def test_positional_targeting_excludes_same_class_component_at_other_mount():
    block = _tiny_block()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert block.ln1._rule_active is True
        assert block.ln2._rule_active is True
        assert block.q_norm._rule_active is False
        assert set(coverage.installed) == {"ln1", "ln2"}
    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False


def test_unsupported_component_at_targeted_mount_is_skipped():
    block = _tiny_block()
    block.ln2 = _PlainMount()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert isinstance(coverage, RelevanceRuleCoverage)
        assert set(coverage.installed) == {"ln1"}
        assert set(coverage.skipped) == {"ln2"}


def test_no_rules_requested_installs_nothing():
    block = _tiny_block()
    with use_relevance_rules(block, RelevanceRules()) as coverage:
        assert coverage.installed == ()
        assert coverage.skipped == ()
        assert block.ln1._rule_active is False
        assert block.ln2._rule_active is False


def test_ordinary_gradients_restored_after_exit():
    block = _tiny_block()
    x = torch.randn(2, 3, 4, requires_grad=True)

    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        y_rule = block(x)
        (grad_rule,) = torch.autograd.grad(y_rule.sum(), x)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False

    x_plain = x.detach().clone().requires_grad_(True)
    y_plain = block(x_plain)
    (grad_plain,) = torch.autograd.grad(y_plain.sum(), x_plain)

    # The LN-rule treats the denominator as constant, so its VJP differs from
    # ordinary autodiff through the same division wherever the denominator
    # actually depends on the input -- true for every row of this fixture.
    assert not torch.allclose(grad_rule, grad_plain)

    # A second plain pass confirms the exit left no residual rule state: it must
    # reproduce grad_plain exactly rather than drifting toward grad_rule.
    x_plain_again = x.detach().clone().requires_grad_(True)
    y_plain_again = block(x_plain_again)
    (grad_plain_again,) = torch.autograd.grad(y_plain_again.sum(), x_plain_again)
    torch.testing.assert_close(grad_plain_again, grad_plain)


def test_nested_contexts_restore_outer_state_on_inner_exit():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)

    with use_relevance_rules(block, RelevanceRules(normalization=True)) as outer_coverage:
        assert block.ln1._rule_active is True
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as inner_coverage:
            assert block.ln1._rule_active is True
            assert torch.equal(block(x), baseline)
        # The inner exit must not disable the rule the outer context still needs.
        assert block.ln1._rule_active is True
        assert block.ln2._rule_active is True
        assert torch.equal(block(x), baseline)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False
    assert set(outer_coverage.installed) == {"ln1", "ln2"}
    assert set(inner_coverage.installed) == {"ln1", "ln2"}


def test_exception_inside_context_leaves_no_rule_state():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)

    with pytest.raises(_Boom):
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            assert block.ln1._rule_active is True
            raise _Boom("failure inside the scoped context")

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False
    assert torch.equal(block(x), baseline)


def test_exception_during_nested_context_restores_outer_state():
    block = _tiny_block()
    x = torch.randn(2, 3, 4)
    baseline = block(x)

    with use_relevance_rules(block, RelevanceRules(normalization=True)):
        with pytest.raises(_Boom):
            with use_relevance_rules(block, RelevanceRules(normalization=True)):
                raise _Boom("failure inside the nested scoped context")
        # The outer context is still active after the inner one unwinds.
        assert block.ln1._rule_active is True
        assert torch.equal(block(x), baseline)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False
