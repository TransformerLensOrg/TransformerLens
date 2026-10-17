"""Granite expert correctness on MPS with fixed routes and an independent oracle."""

import copy
import warnings
from types import SimpleNamespace

import pytest
import torch
import transformers
from packaging.version import Version
from torch import nn
from torch.nn import functional as F
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridExperts,
)

from transformer_lens.model_bridge.generalized_components.moe import MoEBridge

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


class _FixedRoutes(nn.Module):
    def __init__(self):
        super().__init__()
        config = SimpleNamespace(
            hidden_size=64, intermediate_size=32, num_local_experts=4, hidden_act="silu"
        )
        config._experts_implementation = "grouped_mm"
        self.experts = GraniteMoeHybridExperts(config)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in self.experts.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
        self.register_buffer(
            "indices", torch.tensor([[0], [0], [1], [1], [1], [3], [3], [3], [3], [3]])
        )
        self.register_buffer("weights", torch.linspace(0.2, 1.0, 10).unsqueeze(1))

    def forward(self, layer_input, top_k_weights=None):
        weights = self.weights if top_k_weights is None else top_k_weights
        return self.experts(layer_input, self.indices, weights)


def _oracle(block, inputs, top_k_weights=None):
    contributions = []
    route_weights = block.weights if top_k_weights is None else top_k_weights
    for token, indices, weights in zip(inputs, block.indices, route_weights):
        routes = []
        for index, weight in zip(indices, weights):
            gate, up = F.linear(token, block.experts.gate_up_proj[index]).chunk(2, dim=-1)
            routes.append(F.linear(F.silu(gate) * up, block.experts.down_proj[index]) * weight)
        contributions.append(torch.stack(routes).sum(0))
    return torch.stack(contributions)


def test_fixed_route_experts_match_independent_oracle_and_repeat():
    block = _FixedRoutes()
    inputs = torch.randn(10, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = _oracle(block, inputs)
    block.to("mps")
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    observed = []
    handle = block.experts.register_forward_hook(
        lambda module, args, output: observed.append(output)
    )
    try:
        with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = bridge(inputs.to("mps"))
            second = bridge(layer_input=inputs.to("mps"))
        assert first.device.type == "mps"
        torch.testing.assert_close(first.cpu(), expected, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        if (
            Version(torch.__version__).public == "2.11.0"
            and Version(transformers.__version__).public == "5.13.0"
        ):
            assert any("eager Granite" in str(warning.message) for warning in caught)
        assert len(observed) == 2
        assert block.experts.config._experts_implementation == "grouped_mm"
    finally:
        handle.remove()


def test_dispatch_follows_cpu_mps_cpu_migration():
    block = _FixedRoutes()
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    inputs = torch.randn(10, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cpu_before = bridge(inputs)
    assert not caught
    expected = _oracle(block, inputs)
    torch.testing.assert_close(cpu_before, expected, rtol=1e-4, atol=1e-6)
    bridge.to("mps")
    with torch.no_grad():
        metal = bridge(inputs.to("mps"))
    assert metal.device.type == "mps"
    torch.testing.assert_close(metal.cpu(), expected, rtol=1e-4, atol=1e-6)
    bridge.to("cpu")
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cpu_after = bridge(inputs)
    assert not caught
    torch.testing.assert_close(cpu_after, cpu_before, rtol=0, atol=0)
    assert block.experts.config._experts_implementation == "grouped_mm"


def test_explicit_eager_policy_and_expert_hooks_are_preserved():
    block = _FixedRoutes().to("mps")
    block.experts.config._experts_implementation = "eager"
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    inputs = torch.randn(10, 64).to("mps")
    indices, weights = block.indices.clone(), block.weights.clone()
    calls = []

    def observe(module, args, output):
        calls.append(args)
        return output * 2

    handle = block.experts.register_forward_hook(observe)
    try:
        with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected = _oracle(block, inputs) * 2
            actual = bridge(inputs)
        assert not caught
        assert len(calls) == 1
        torch.testing.assert_close(calls[0][0], inputs, rtol=0, atol=0)
        torch.testing.assert_close(calls[0][1], indices, rtol=0, atol=0)
        torch.testing.assert_close(calls[0][2], weights, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        assert actual.shape == inputs.shape
        assert block.experts.config._experts_implementation == "eager"
        torch.testing.assert_close(block.indices, indices, rtol=0, atol=0)
        torch.testing.assert_close(block.weights, weights, rtol=0, atol=0)
    finally:
        handle.remove()


def test_corrected_experts_preserve_input_route_and_parameter_gradients():
    block = _FixedRoutes()
    block.indices = torch.tensor(
        [
            [0, 1],
            [0, 3],
            [1, 0],
            [1, 3],
            [1, 0],
            [3, 0],
            [3, 1],
            [3, 0],
            [3, 1],
            [3, 0],
        ]
    )
    reference = copy.deepcopy(block)
    block.to("mps")
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    data = torch.randn(10, 64, generator=torch.Generator().manual_seed(1))
    cpu_input = data.clone().requires_grad_()
    mps_input = data.to("mps").requires_grad_()
    cpu_routes = torch.linspace(0.2, 1.0, 20).reshape(10, 2).requires_grad_()
    mps_routes = torch.linspace(0.2, 1.0, 20, device="mps").reshape(10, 2).requires_grad_()
    calls = []

    def observe(module, args, output):
        calls.append(args)
        return output * 2

    handle = block.experts.register_forward_hook(observe)
    try:
        expected = _oracle(reference, cpu_input, cpu_routes) * 2
        actual = bridge(mps_input, top_k_weights=mps_routes)
    finally:
        handle.remove()
    assert len(calls) == 1
    assert calls[0][0] is mps_input
    assert calls[0][1] is block.indices
    assert calls[0][2] is mps_routes
    assert actual.shape == mps_input.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-6)
    cpu_targets = (cpu_input, cpu_routes, *reference.experts.parameters())
    mps_targets = (mps_input, mps_routes, *block.experts.parameters())
    cpu_grads = torch.autograd.grad(expected.square().sum(), cpu_targets)
    mps_grads = torch.autograd.grad(actual.square().sum(), mps_targets)
    for measured, oracle in zip(mps_grads, cpu_grads):
        assert measured.device.type == "mps"
        torch.testing.assert_close(measured.cpu(), oracle, rtol=1e-4, atol=1e-6)
    assert cpu_grads[1].abs().max() > 0
    assert all(gradient.abs().max() > 0 for gradient in cpu_grads[2:])
    torch.testing.assert_close(block.indices.cpu(), reference.indices, rtol=0, atol=0)
    for parameter, saved in zip(block.experts.parameters(), reference.experts.parameters()):
        torch.testing.assert_close(parameter.cpu(), saved, rtol=0, atol=0)
    assert block.experts.config._experts_implementation == "grouped_mm"


def test_failed_expert_forward_leaves_policy_and_parameters_intact():
    block = _FixedRoutes().to("mps")
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    inputs = torch.randn(10, 64).to("mps")
    parameters = [parameter.clone() for parameter in block.experts.parameters()]
    activation = block.experts.act_fn

    class FailingActivation(nn.Module):
        def forward(self, value):
            raise ValueError("expert activation failed")

    block.experts.act_fn = FailingActivation()
    try:
        with pytest.raises(ValueError, match="expert activation failed"):
            bridge(inputs)
    finally:
        block.experts.act_fn = activation
    assert block.experts.config._experts_implementation == "grouped_mm"
    with torch.no_grad():
        actual = bridge(inputs)
        expected = _oracle(block, inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    for parameter, saved in zip(block.experts.parameters(), parameters):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
