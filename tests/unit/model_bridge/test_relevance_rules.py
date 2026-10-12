"""Scoped relevance rules: kind-aware coverage, positional targeting, cleanup, and nesting."""

import dataclasses

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleCoverage,
    RelevanceRules,
    RelevanceRuleUnsupportedError,
    _RelevanceRuleCoverageEntry,
    ln_rule,
    use_relevance_rules,
)


def _normalization_entries(*paths):
    return tuple(_RelevanceRuleCoverageEntry(kind="normalization", path=path) for path in paths)


class _FakeNormComponent(nn.Module):
    """A minimal ln1/ln2-style target: installs the real LN-rule primitive on request."""

    _relevance_rule_kinds = ("normalization",)

    def __init__(self, eps: float = 1e-2):
        super().__init__()
        # The LN-rule correction shrinks linearly with eps here, so a realistic 1e-6 would
        # leave it too small for torch.allclose to detect end to end.
        self.eps = eps
        self._rule_active = False

    def _enable_relevance_rule(self, kind: str) -> None:
        self._rule_active = True

    def _disable_relevance_rule(self, kind: str) -> None:
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
    """Block with ln1/ln2, an MLP, and a same-class q_norm that positional targeting must skip."""

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


class _SandwichBlock(nn.Module):
    """Sandwich-norm block: pre-norms ln1/ln2 plus post-norms ln1_post/ln2_post."""

    def __init__(self):
        super().__init__()
        self.ln1: nn.Module = _FakeNormComponent()
        self.ln1_post: nn.Module = _FakeNormComponent()
        self.ln2: nn.Module = _FakeNormComponent()
        self.ln2_post: nn.Module = _FakeNormComponent()
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.ln1_post(x)
        x = self.ln2(x)
        x = self.ln2_post(x)
        return self.mlp(x)


class _FakeGatedMLPComponent(nn.Module):
    """mlp-mount target tracking the activation and multiplicative_gate kinds independently."""

    _relevance_rule_kinds = ("activation", "multiplicative_gate")

    def __init__(self):
        super().__init__()
        self._activation_rule_active = False
        self._gate_rule_active = False

    def _enable_relevance_rule(self, kind: str) -> None:
        if kind == "activation":
            self._activation_rule_active = True
        elif kind == "multiplicative_gate":
            self._gate_rule_active = True

    def _disable_relevance_rule(self, kind: str) -> None:
        if kind == "activation":
            self._activation_rule_active = False
        elif kind == "multiplicative_gate":
            self._gate_rule_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
        assert coverage.installed == _normalization_entries("ln1", "ln2")
    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False


def test_normalization_rule_installs_on_sandwich_post_norm_mounts():
    block = _SandwichBlock()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert block.ln1._rule_active is True
        assert block.ln1_post._rule_active is True
        assert block.ln2._rule_active is True
        assert block.ln2_post._rule_active is True
        assert coverage.installed == _normalization_entries("ln1", "ln1_post", "ln2", "ln2_post")
        assert coverage.skipped == ()
    assert block.ln1_post._rule_active is False
    assert block.ln2_post._rule_active is False


def test_sandwich_post_norm_on_python_path_is_reported_skipped_never_absent():
    block = _SandwichBlock()
    block.ln2_post = _PlainMount()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert coverage.installed == _normalization_entries("ln1", "ln1_post", "ln2")
        assert coverage.skipped == _normalization_entries("ln2_post")


def test_requesting_a_kind_with_no_canonical_mount_raises():
    block = _tiny_block()
    with pytest.raises(ValueError, match="attention"):
        with use_relevance_rules(block, RelevanceRules(attention=True)):
            pass


def test_unsupported_component_at_targeted_mount_is_skipped():
    block = _tiny_block()
    block.ln2 = _PlainMount()
    with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
        assert isinstance(coverage, RelevanceRuleCoverage)
        assert coverage.installed == _normalization_entries("ln1")
        assert coverage.skipped == _normalization_entries("ln2")


def test_requesting_unsupported_kind_on_capable_component_raises():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    block.mlp._relevance_rule_kinds = ("activation",)
    block.mlp._relevance_rule_unsupported_kinds = ("multiplicative_gate",)
    with pytest.raises(RelevanceRuleUnsupportedError, match="mlp"):
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            pass
    assert block.mlp._gate_rule_active is False


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

    assert not torch.allclose(grad_rule, grad_plain)

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
        assert block.ln1._rule_active is True
        assert block.ln2._rule_active is True
        assert torch.equal(block(x), baseline)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False
    assert outer_coverage.installed == _normalization_entries("ln1", "ln2")
    assert inner_coverage.installed == _normalization_entries("ln1", "ln2")


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
        assert block.ln1._rule_active is True
        assert torch.equal(block(x), baseline)

    assert block.ln1._rule_active is False
    assert block.ln2._rule_active is False


@pytest.mark.parametrize("kind", ["activation", "multiplicative_gate"])
def test_one_kind_requested_on_a_two_kind_mount_leaves_the_other_kind_inactive(kind):
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    with use_relevance_rules(block, RelevanceRules(**{kind: True})) as coverage:
        assert block.mlp._gate_rule_active is (kind == "multiplicative_gate")
        assert block.mlp._activation_rule_active is (kind == "activation")
        assert [(entry.kind, entry.path) for entry in coverage.installed] == [(kind, "mlp")]
        assert coverage.skipped == ()
    assert block.mlp._gate_rule_active is False
    assert block.mlp._activation_rule_active is False


@pytest.mark.parametrize("supported_kind", ["activation", "multiplicative_gate"])
def test_kind_coverage_distinguishes_installed_and_skipped_at_the_same_path(supported_kind):
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    block.mlp._relevance_rule_kinds = (supported_kind,)
    other_kind = "multiplicative_gate" if supported_kind == "activation" else "activation"
    with use_relevance_rules(
        block, RelevanceRules(activation=True, multiplicative_gate=True)
    ) as coverage:
        assert [(entry.kind, entry.path) for entry in coverage.installed] == [
            (supported_kind, "mlp")
        ]
        assert [(entry.kind, entry.path) for entry in coverage.skipped] == [(other_kind, "mlp")]


def test_aliases_do_not_duplicate_kind_coverage():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    block.original = nn.Module()
    block.original.mlp = block.mlp
    with use_relevance_rules(
        block, RelevanceRules(activation=True, multiplicative_gate=True)
    ) as coverage:
        assert [(entry.kind, entry.path) for entry in coverage.installed] == [
            ("activation", "mlp"),
            ("multiplicative_gate", "mlp"),
        ]
        assert coverage.skipped == ()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(coverage.installed[0], "kind", "normalization")


def test_both_kinds_requested_together_both_activate_on_the_same_mount():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    with use_relevance_rules(
        block, RelevanceRules(activation=True, multiplicative_gate=True)
    ) as coverage:
        assert block.mlp._activation_rule_active is True
        assert block.mlp._gate_rule_active is True
        assert [(entry.kind, entry.path) for entry in coverage.installed] == [
            ("activation", "mlp"),
            ("multiplicative_gate", "mlp"),
        ]
    assert block.mlp._activation_rule_active is False
    assert block.mlp._gate_rule_active is False


def test_nested_scopes_over_different_kinds_on_one_mount_refcount_independently():
    block = _tiny_block()
    block.mlp = _FakeGatedMLPComponent()
    with use_relevance_rules(block, RelevanceRules(activation=True)):
        assert block.mlp._activation_rule_active is True
        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            assert block.mlp._activation_rule_active is True
            assert block.mlp._gate_rule_active is True
        assert block.mlp._activation_rule_active is True
        assert block.mlp._gate_rule_active is False
    assert block.mlp._activation_rule_active is False
