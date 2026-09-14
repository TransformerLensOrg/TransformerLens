"""LN-rule integration on NormalizationBridge's native-autograd path.

Covers the commit-5 contract: the rule-wrapped native forward is bit-identical to
today's native forward by construction (the wrapping calls ``original_component(x)``
itself rather than reproducing its numerics), the backward follows the LN-rule
(denominator treated as constant) while weight/bias keep their ordinary gradient,
targeting is positional so an ln1/ln2 mount that never reaches the native-autograd
branch reports as skipped rather than silently leaving ordinary gradients in place,
and a hook that would otherwise silently fall back instead raises while the rule is
active.
"""


import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge._relevance_rules import (
    RelevanceRuleConflictError,
    RelevanceRules,
    use_relevance_rules,
)
from transformer_lens.model_bridge.generalized_components.normalization import (
    LayerNormPreBridge,
    NormalizationBridge,
    RMSNormPreBridge,
)


class _Cfg:
    def __init__(
        self,
        uses_rms_norm: bool = False,
        eps: float = 1e-5,
        rmsnorm_uses_offset: bool = False,
        layer_norm_folding: bool = False,
    ):
        self.uses_rms_norm = uses_rms_norm
        self.eps = eps
        self.rmsnorm_uses_offset = rmsnorm_uses_offset
        self.layer_norm_folding = layer_norm_folding


class _TinyRMSNorm(nn.Module):
    """Minimal RMSNorm mirroring LlamaRMSNorm's forward."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1 + 1.0)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.variance_epsilon)


class _TinyGemmaRMSNorm(nn.Module):
    """Minimal Gemma-style RMSNorm: weight is stored as an offset from 1."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.variance_epsilon)
        return x_normed * (1.0 + self.weight)


def _layernorm(d: int) -> nn.LayerNorm:
    layer = nn.LayerNorm(d, eps=1e-5)
    nn.init.normal_(layer.weight, std=0.1)
    nn.init.normal_(layer.bias, std=0.1)
    return layer


class _Block(nn.Module):
    """Mounts a normalization bridge at the canonical ln1 position."""

    def __init__(self, norm: NormalizationBridge):
        super().__init__()
        self.ln1 = norm


def _make_bridge(
    native: bool,
    rms: bool = False,
    offset: bool = False,
    d: int = 16,
    layer_norm_folding: bool = False,
) -> NormalizationBridge:
    layer: nn.Module
    if offset:
        layer = _TinyGemmaRMSNorm(d)
    elif rms:
        layer = _TinyRMSNorm(d)
    else:
        layer = _layernorm(d)
    bridge = NormalizationBridge(
        name="ln1",
        config=_Cfg(
            uses_rms_norm=rms or offset,
            rmsnorm_uses_offset=offset,
            layer_norm_folding=layer_norm_folding,
        ),
        use_native_layernorm_autograd=native,
    )
    bridge.set_original_component(layer)
    return bridge


def _denom_detached_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    uses_rms: bool,
    offset: bool,
    eps: float,
) -> torch.Tensor:
    """Manual recompute with the denominator detached: the LN-rule's target VJP."""
    x_centered = x if uses_rms else x - x.mean(-1, keepdim=True)
    denom = (x_centered.pow(2).mean(-1, keepdim=True) + eps).sqrt().detach()
    w_eff = (1.0 + weight) if offset else weight
    out = (x_centered / denom) * w_eff
    if bias is not None:
        out = out + bias
    return out


class TestForwardIdentity:
    @pytest.mark.parametrize("rms", [False, True], ids=["layernorm", "rmsnorm"])
    def test_active_forward_matches_inactive_forward(self, rms):
        bridge = _make_bridge(native=True, rms=rms)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        baseline = bridge(x)
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            active = bridge(x)
        assert torch.equal(active, baseline)

    def test_active_forward_matches_original_component_directly(self):
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            active = bridge(x)
        assert torch.equal(active, bridge.original_component(x))


class TestVJPMatchesDetachedDenomOracle:
    def test_layernorm(self):
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            y = bridge(x)
            y.sum().backward()
        grad_rule = x.grad.clone()

        x_oracle = x.detach().clone().requires_grad_(True)
        y_oracle = _denom_detached_oracle(
            x_oracle,
            bridge.original_component.weight,
            bridge.original_component.bias,
            uses_rms=False,
            offset=False,
            eps=1e-5,
        )
        y_oracle.sum().backward()
        torch.testing.assert_close(grad_rule, x_oracle.grad)

    @pytest.mark.parametrize("offset", [False, True], ids=["plain_rms", "gemma_offset"])
    def test_rmsnorm(self, offset):
        bridge = _make_bridge(native=True, rms=not offset, offset=offset)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            y = bridge(x)
            y.sum().backward()
        grad_rule = x.grad.clone()

        x_oracle = x.detach().clone().requires_grad_(True)
        y_oracle = _denom_detached_oracle(
            x_oracle,
            bridge.original_component.weight,
            None,
            uses_rms=True,
            offset=offset,
            eps=1e-5,
        )
        y_oracle.sum().backward()
        torch.testing.assert_close(grad_rule, x_oracle.grad)

    def test_rule_grad_disagrees_with_ordinary_autodiff(self):
        """The whole point of the rule: it must actually change the gradient."""
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16, requires_grad=True)

        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            y_rule = bridge(x)
            (grad_rule,) = torch.autograd.grad(y_rule.sum(), x)

        x_plain = x.detach().clone().requires_grad_(True)
        y_plain = bridge(x_plain)
        (grad_plain,) = torch.autograd.grad(y_plain.sum(), x_plain)
        assert not torch.allclose(grad_rule, grad_plain)


class TestParameterGradientsPreserved:
    def test_weight_and_bias_receive_ordinary_gradient(self):
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16, requires_grad=True)
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            y = bridge(x)
            y.sum().backward()
        weight_grad = bridge.original_component.weight.grad
        bias_grad = bridge.original_component.bias.grad
        assert weight_grad is not None and torch.isfinite(weight_grad).all()
        assert bias_grad is not None and torch.isfinite(bias_grad).all()

        # Ordinary gradient: d(output)/d(weight) = normalized value (pre-weight),
        # independent of the rule's treatment of the x-path; d(output)/d(bias) = 1.
        x_oracle = x.detach().clone()
        normalized = _denom_detached_oracle(
            x_oracle,
            torch.ones_like(bridge.original_component.weight),
            None,
            uses_rms=False,
            offset=False,
            eps=1e-5,
        )
        reduce_dims = tuple(range(normalized.dim() - 1))
        expected_weight_grad = normalized.sum(dim=reduce_dims)
        expected_bias_grad = torch.full_like(bias_grad, x.shape[0] * x.shape[1])
        torch.testing.assert_close(weight_grad, expected_weight_grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(bias_grad, expected_bias_grad, atol=1e-4, rtol=1e-4)


class TestRuleInactiveRegression:
    @pytest.mark.parametrize("native", [True, False], ids=["native_autograd", "python_norm"])
    def test_forward_byte_identical_to_ordinary_call(self, native):
        bridge = _make_bridge(native=native)
        x = torch.randn(2, 5, 16)
        expected = bridge(x)
        # No use_relevance_rules context at all: today's behavior, unconditionally.
        actual = bridge(x)
        assert torch.equal(actual, expected)

    def test_non_native_path_rule_request_is_skipped_and_forward_unaffected(self):
        """A python-norm-path bridge (no native autograd, no folding) never reaches
        the branch the LN-rule wraps, so it must be reported skipped rather than
        silently leaving ordinary gradients in place under a claimed rule."""
        bridge = _make_bridge(native=False)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        baseline = bridge(x)
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
            assert coverage.installed == ()
            assert coverage.skipped == ("ln1",)
            active = bridge(x)
        assert torch.equal(active, baseline)

    def test_layer_norm_folding_config_flag_makes_the_rule_installable(self):
        """layer_norm_folding also dispatches through the native-autograd branch,
        so the rule must be installable there even with use_native_layernorm_autograd
        left False."""
        bridge = _make_bridge(native=False, layer_norm_folding=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        baseline = bridge(x)
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
            assert coverage.installed == ("ln1",)
            active = bridge(x)
        assert torch.equal(active, baseline)

    @pytest.mark.parametrize(
        "bridge_cls", [LayerNormPreBridge, RMSNormPreBridge], ids=["ln_pre", "rms_pre"]
    )
    def test_param_free_pre_norm_is_skipped_and_forward_unaffected(self, bridge_cls):
        """LNPre/RMSPre always take the python-norm path regardless of the native
        flag, so they must never be reported installed."""
        bridge = bridge_cls(name="ln1", config=_Cfg())
        bridge.set_original_component(nn.LayerNorm(16, eps=1e-5, elementwise_affine=False))
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        baseline = bridge(x)
        with use_relevance_rules(block, RelevanceRules(normalization=True)) as coverage:
            assert coverage.installed == ()
            assert coverage.skipped == ("ln1",)
            active = bridge(x)
        assert torch.equal(active, baseline)


class TestFailClosedHookPrecedence:
    def test_bwd_hook_raises_while_rule_active(self):
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16, requires_grad=True)
        bridge.hook_normalized.add_hook(lambda g, hook=None: g, dir="bwd")
        with pytest.raises(RelevanceRuleConflictError, match="Backward hooks"):
            with use_relevance_rules(block, RelevanceRules(normalization=True)):
                bridge(x)
        bridge.hook_normalized.remove_hooks()

    def test_forward_edit_raises_while_rule_active(self):
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        bridge.hook_scale.add_hook(lambda t, hook=None: t * 2.0)
        with pytest.raises(RelevanceRuleConflictError, match="forward hook edited"):
            with use_relevance_rules(block, RelevanceRules(normalization=True)):
                bridge(x)
        bridge.hook_scale.remove_hooks()

    def test_observation_only_forward_hook_does_not_raise_while_rule_active(self):
        """A hook that only observes (returns None) is not an edit and must not
        trip the fail-closed check."""
        bridge = _make_bridge(native=True)
        block = _Block(bridge)
        x = torch.randn(2, 5, 16)
        baseline = bridge(x)
        cache = {}

        def observe(tensor, hook=None):
            cache["normalized"] = tensor.detach()
            return None

        bridge.hook_normalized.add_hook(observe)
        with use_relevance_rules(block, RelevanceRules(normalization=True)):
            active = bridge(x)
        bridge.hook_normalized.remove_hooks()
        assert torch.equal(active, baseline)
        assert "normalized" in cache

    def test_bwd_hook_still_falls_back_with_warning_when_rule_inactive(self):
        """Control: unchanged behavior when no rule is active (regression guard)."""
        bridge = _make_bridge(native=True)
        x = torch.randn(2, 5, 16, requires_grad=True)
        bridge.hook_normalized.add_hook(lambda g, hook=None: g, dir="bwd")
        with pytest.warns(UserWarning, match="Backward hooks"):
            bridge(x)
        bridge.hook_normalized.remove_hooks()

    def test_edit_still_falls_back_with_warning_when_rule_inactive(self):
        """Control: unchanged behavior when no rule is active (regression guard)."""
        bridge = _make_bridge(native=True)
        x = torch.randn(2, 5, 16)
        bridge.hook_scale.add_hook(lambda t, hook=None: t * 2.0)
        with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
            bridge(x)
        bridge.hook_scale.remove_hooks()
