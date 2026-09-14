"""Identity-/Half-rule integration on JointGateUpMLPBridge's reconstructed forward.

Unlike the raw GatedMLPBridge path (recompute-from-weights, tested separately),
JointGateUpMLPBridge already reconstructs its forward in Python as
``act_fn(gate_output) * up_output`` through separate gate/up LinearBridge
submodules, so the rules attach directly at that multiplication -- no
weights-recompute Function is needed here. This covers that the reconstructed
forward is unaffected by an inactive rule, that both rules apply correctly
together and independently, and that gradients match the oracle produced by the
already-tested Identity-/Half-rule primitives directly.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRules,
    half_rule,
    identity_rule,
    use_relevance_rules,
)
from transformer_lens.model_bridge.generalized_components.joint_gate_up_mlp import (
    JointGateUpMLPBridge,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class _Cfg:
    hidden_act = "silu"


class _TinyPhi3MLP(nn.Module):
    """Mirrors Phi-3/GLM's fused gate_up_proj structure."""

    def __init__(self, d_model: int = 4, d_mlp: int = 8, bias: bool = False):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_mlp, bias=bias)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=bias)
        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.activation_fn(gate) * up)


class _Block(nn.Module):
    """Mounts a joint gate-up bridge at the canonical mlp position."""

    def __init__(self, mlp: JointGateUpMLPBridge):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _make_bridge(bias: bool = False) -> tuple[_Block, _TinyPhi3MLP]:
    torch.manual_seed(0)
    hf_mlp = _TinyPhi3MLP(bias=bias)
    bridge = JointGateUpMLPBridge(name="mlp", config=_Cfg(), submodules={})
    out_bridge = LinearBridge(name="down_proj")
    bridge.add_module("out", out_bridge)
    bridge.set_original_component(hf_mlp)
    out_bridge.set_original_component(hf_mlp.down_proj)
    return _Block(bridge), hf_mlp


def _oracle_grads(hf_mlp, gate_proj, up_proj, x, activation_active: bool, gate_active: bool):
    for p in hf_mlp.parameters():
        p.grad = None
    x_oracle = x.detach().clone().requires_grad_(True)
    gate_output = F.linear(x_oracle, gate_proj.weight, gate_proj.bias)
    up_output = F.linear(x_oracle, up_proj.weight, up_proj.bias)
    activated = identity_rule(gate_output, F.silu) if activation_active else F.silu(gate_output)
    gated = half_rule(activated, up_output) if gate_active else activated * up_output
    down = hf_mlp.down_proj(gated)
    down.sum().backward()
    return x_oracle.grad.clone()


class TestJointGateUpRelevanceRuleCapability:
    def test_capable_of_both_kinds(self):
        block, _ = _make_bridge()
        assert set(block.mlp._relevance_rule_kinds) == {"activation", "multiplicative_gate"}


class TestJointGateUpRelevanceRuleForwardIdentity:
    def test_forward_identical_while_rule_active(self):
        block, _ = _make_bridge()
        x = torch.randn(3, 4)
        baseline = block(x)
        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            active = block(x)
        assert torch.equal(active, baseline)

    def test_forward_unchanged_when_rule_inactive(self):
        block, _ = _make_bridge()
        x = torch.randn(3, 4)
        before = block(x)
        after = block(x)
        assert torch.equal(before, after)


class TestJointGateUpRelevanceRuleVJP:
    @pytest.mark.parametrize(
        ("activation_active", "gate_active"),
        [(True, True), (True, False), (False, True)],
    )
    def test_matches_manually_composed_oracle(self, activation_active, gate_active):
        block, hf_mlp = _make_bridge()
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(
            block,
            RelevanceRules(
                activation=activation_active,
                multiplicative_gate=gate_active,
            ),
        ):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()

        gate_proj = block.mlp.gate.original_component
        up_proj = getattr(block.mlp, "in").original_component
        expected_grad_x = _oracle_grads(
            hf_mlp, gate_proj, up_proj, x, activation_active, gate_active
        )
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)

    def test_rule_inactive_gradients_are_ordinary(self):
        block, hf_mlp = _make_bridge()
        x = torch.randn(3, 4, requires_grad=True)

        out = block(x)
        out.sum().backward()
        grad_x = x.grad.clone()

        for p in hf_mlp.parameters():
            p.grad = None
        x_plain = x.detach().clone().requires_grad_(True)
        plain = hf_mlp(x_plain)
        plain.sum().backward()

        torch.testing.assert_close(grad_x, x_plain.grad)

    def test_bias_free_and_biased_projections_are_both_handled(self):
        block, hf_mlp = _make_bridge(bias=True)
        x = torch.randn(3, 4, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(activation=True, multiplicative_gate=True)):
            out = block(x)
            out.sum().backward()

        grad_x = x.grad.clone()
        gate_proj = block.mlp.gate.original_component
        up_proj = getattr(block.mlp, "in").original_component
        expected_grad_x = _oracle_grads(hf_mlp, gate_proj, up_proj, x, True, True)
        torch.testing.assert_close(grad_x, expected_grad_x, atol=1e-5, rtol=1e-5)
