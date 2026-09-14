"""Identity-/Half-rule integration on GatedMLPBridge's recompute-from-weights path.

Covers the commit-6 contract for the raw (non-fused) gated-MLP path: the opaque
native forward stays bit-identical to today's forward by construction (the wrapping
returns ``original_component(x)`` itself, never a reproduced numeric value), the
backward recomputes gate/up/down from the TL-oriented ``W_gate``/``W_in``/``W_out``
and applies the Identity-rule to the activation and/or the Half-rule to the gate*up
product per whichever is independently active, weight and bias keep their ordinary
gradient, and the recompute is allowlisted by the underlying HF module class
(``nn.Linear`` and ``Conv1D``) rather than reported installed for an unrecognized
backing module.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    half_rule,
    identity_rule,
    use_relevance_rules,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.gated_mlp import (
    GatedMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class _Cfg:
    hidden_act = "silu"


class _TinyGatedMLP(nn.Module):
    """Mirrors the Qwen2/Llama/Gemma gated-MLP structure: opaque single call."""

    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    """Mounts a gated-MLP bridge at the canonical mlp position."""

    def __init__(self, mlp: GeneralizedComponent):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _make_projections(backing_class: str, d_model: int = 4, d_mlp: int = 8, bias: bool = True):
    torch.manual_seed(0)
    if backing_class == "nn.Linear":
        return (
            nn.Linear(d_model, d_mlp, bias=bias),
            nn.Linear(d_model, d_mlp, bias=bias),
            nn.Linear(d_mlp, d_model, bias=bias),
        )
    return (
        Conv1D(d_mlp, d_model),
        Conv1D(d_mlp, d_model),
        Conv1D(d_model, d_mlp),
    )


def _make_bridge(backing_class: str, bias: bool = True) -> tuple[_Block, _TinyGatedMLP]:
    gate_proj, up_proj, down_proj = _make_projections(backing_class, bias=bias)
    hf_mlp = _TinyGatedMLP(gate_proj, up_proj, down_proj)

    bridge = GatedMLPBridge(name="mlp", config=_Cfg())
    gate_bridge = LinearBridge(name="gate_proj")
    in_bridge = LinearBridge(name="up_proj")
    out_bridge = LinearBridge(name="down_proj")
    bridge.add_module("gate", gate_bridge)
    bridge.add_module("in", in_bridge)
    bridge.add_module("out", out_bridge)
    bridge.set_original_component(hf_mlp)
    gate_bridge.set_original_component(gate_proj)
    in_bridge.set_original_component(up_proj)
    out_bridge.set_original_component(down_proj)

    return _Block(bridge), hf_mlp


BACKING_CLASSES = ["nn.Linear", "Conv1D"]


class TestGatedMLPRelevanceRuleCapability:
    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_capable_of_both_kinds_when_allowlisted(self, backing_class):
        block, _ = _make_bridge(backing_class)
        assert set(block.mlp._relevance_rule_kinds) == {"activation", "multiplicative_gate"}

    def test_not_capable_when_backing_module_is_unrecognized(self):
        class _OpaqueProj(nn.Module):
            """Neither nn.Linear nor Conv1D -- an unrecognized backing class."""

            def __init__(self, d_in: int, d_out: int):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(d_out, d_in))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight.T

        gate_proj = _OpaqueProj(4, 8)
        up_proj = _OpaqueProj(4, 8)
        down_proj = _OpaqueProj(8, 4)
        hf_mlp = _TinyGatedMLP(gate_proj, up_proj, down_proj)

        bridge = GatedMLPBridge(name="mlp", config=_Cfg())
        gate_bridge = LinearBridge(name="gate_proj")
        in_bridge = LinearBridge(name="up_proj")
        out_bridge = LinearBridge(name="down_proj")
        bridge.add_module("gate", gate_bridge)
        bridge.add_module("in", in_bridge)
        bridge.add_module("out", out_bridge)
        bridge.set_original_component(hf_mlp)
        gate_bridge.set_original_component(gate_proj)
        in_bridge.set_original_component(up_proj)
        out_bridge.set_original_component(down_proj)
        block = _Block(bridge)

        assert bridge._relevance_rule_kinds == ()

        x = torch.randn(2, 4, requires_grad=True)
        baseline = block(x)
        with use_relevance_rules(
            block, RelevanceRules(activation=True, multiplicative_gate=True)
        ) as coverage:
            assert set(coverage.skipped) == {"mlp"}
            assert coverage.installed == ()
            active = block(x)
            assert torch.equal(active, baseline)


class TestGatedMLPRelevanceRuleForwardIdentity:
    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_forward_identical_while_rule_active(self, backing_class):
        block, _ = _make_bridge(backing_class)
        x = torch.randn(3, 4)
        baseline = block(x)
        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            active = block(x)
        assert torch.equal(active, baseline)

    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_forward_unchanged_when_rule_inactive(self, backing_class):
        block, _ = _make_bridge(backing_class)
        x = torch.randn(3, 4)
        before = block(x)
        after = block(x)
        assert torch.equal(before, after)


class TestGatedMLPRelevanceRuleVJP:
    def _oracle_grads(self, hf_mlp, x, activation_active, gate_active):
        # hf_mlp's own parameters are shared with the bridge under test, so a prior
        # backward through the bridge already left gradients on them; without
        # resetting first, this second backward would accumulate on top instead of
        # producing an independently comparable oracle.
        for p in hf_mlp.parameters():
            p.grad = None
        x_oracle = x.detach().clone().requires_grad_(True)
        gate_output = hf_mlp.gate_proj(x_oracle)
        up_output = hf_mlp.up_proj(x_oracle)
        activated = identity_rule(gate_output, F.silu) if activation_active else F.silu(gate_output)
        gated = half_rule(activated, up_output) if gate_active else activated * up_output
        down = hf_mlp.down_proj(gated)
        down.sum().backward()
        grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}
        return x_oracle.grad.clone(), grads

    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_both_rules_active_matches_manually_composed_oracle(self, backing_class):
        block, hf_mlp = _make_bridge(backing_class)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}

        expected_grad_x, expected_grads = self._oracle_grads(
            hf_mlp, x, activation_active=True, gate_active=True
        )
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
        for name, grad in grads.items():
            torch.testing.assert_close(grad, expected_grads[name], atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_activation_only_leaves_gate_split_ordinary(self, backing_class):
        block, hf_mlp = _make_bridge(backing_class)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        expected_grad_x, _ = self._oracle_grads(
            hf_mlp, x, activation_active=True, gate_active=False
        )
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("backing_class", BACKING_CLASSES)
    def test_multiplicative_gate_only_leaves_activation_ordinary(self, backing_class):
        block, hf_mlp = _make_bridge(backing_class)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        expected_grad_x, _ = self._oracle_grads(
            hf_mlp, x, activation_active=False, gate_active=True
        )
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)

    def test_rule_inactive_gradients_are_ordinary(self):
        block, hf_mlp = _make_bridge("nn.Linear")
        x = torch.randn(3, 4, requires_grad=True)

        out = block(x)
        out.sum().backward()
        grad_x = x.grad.clone()
        grads = {n: p.grad.clone() for n, p in hf_mlp.named_parameters()}

        for p in hf_mlp.parameters():
            p.grad = None
        x_plain = x.detach().clone().requires_grad_(True)
        plain = hf_mlp(x_plain)
        plain.sum().backward()

        torch.testing.assert_close(grad_x, x_plain.grad)
        for name, grad in grads.items():
            torch.testing.assert_close(grad, dict(hf_mlp.named_parameters())[name].grad)

    def test_bias_free_projections_are_handled(self):
        block, hf_mlp = _make_bridge("nn.Linear", bias=False)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        expected_grad_x, expected_grads = self._oracle_grads(
            hf_mlp, x, activation_active=True, gate_active=True
        )
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
        for name, param in hf_mlp.named_parameters():
            torch.testing.assert_close(param.grad, expected_grads[name], atol=1e-5, rtol=1e-5)
