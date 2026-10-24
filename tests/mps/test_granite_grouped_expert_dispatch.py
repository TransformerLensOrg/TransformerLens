"""Shared HF expert correctness on MPS with fixed routes and an independent oracle."""

import copy
import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers.integrations import moe as hf_moe
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridExperts,
)
from transformers.models.mixtral.modeling_mixtral import MixtralExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from transformer_lens.model_bridge.generalized_components import moe
from transformer_lens.model_bridge.generalized_components.moe import (
    MoEBridge,
    _mps_output_slice_writes_are_broken,
)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


@pytest.fixture(
    params=[GraniteMoeHybridExperts, GraniteMoeExperts, MixtralExperts, Qwen3MoeExperts]
)
def expert_class(request):
    return request.param


@pytest.fixture(autouse=True)
def isolated_probe_cache():
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()
    moe._MPS_EXPERTS_WARNED = False
    yield
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()
    moe._MPS_EXPERTS_WARNED = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_runtime_probe_matches_direct_slice_write_diagnostic(dtype):
    generator = torch.Generator(device="cpu").manual_seed(0)
    inputs = torch.randn(10, 64, generator=generator).to(device="mps", dtype=dtype)
    weights = torch.randn(4, 64, 32, generator=generator).to(device="mps", dtype=dtype)
    with torch.no_grad():
        expected = inputs[5:] @ weights[3]
        output = torch.zeros(10, 32, device="mps", dtype=dtype)
        torch.mm(inputs[5:], weights[3], out=output[5:])
    products_match = torch.allclose(
        output[5:], expected, rtol=max(1e-5, 2 * torch.finfo(dtype).eps), atol=1e-5
    )
    prefix_intact = torch.count_nonzero(output[:5]).item() == 0
    cpu_rng = torch.get_rng_state().clone()
    mps_rng = torch.mps.get_rng_state().clone()
    detected = _mps_output_slice_writes_are_broken(inputs.device, dtype)
    assert detected == (not (products_match and prefix_intact))
    assert _mps_output_slice_writes_are_broken(inputs.device, dtype) == detected
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.mps.get_rng_state(), mps_rng, rtol=0, atol=0)


class _FixedRoutes(nn.Module):
    def __init__(self, expert_class=GraniteMoeHybridExperts):
        super().__init__()
        config = SimpleNamespace(
            hidden_size=64,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_local_experts=4,
            num_experts=4,
            hidden_act="silu",
        )
        config._experts_implementation = "grouped_mm"
        self.experts = expert_class(config)
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


def test_fixed_route_experts_match_independent_oracle_and_repeat(expert_class):
    block = _FixedRoutes(expert_class)
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
        fallback_warnings = [
            warning for warning in caught if "eager experts" in str(warning.message)
        ]
        assert len(fallback_warnings) == int(
            _mps_output_slice_writes_are_broken(first.device, first.dtype)
        )
        assert len(observed) == 2
        assert block.experts.config._experts_implementation == "grouped_mm"
    finally:
        handle.remove()


def test_dispatch_follows_cpu_mps_cpu_migration(expert_class):
    block = _FixedRoutes(expert_class)
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


def test_explicit_eager_policy_and_expert_hooks_are_preserved(expert_class):
    block = _FixedRoutes(expert_class).to("mps")
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


def test_corrected_experts_preserve_input_route_and_parameter_gradients(expert_class):
    block = _FixedRoutes(expert_class)
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


def test_failed_expert_forward_leaves_policy_and_parameters_intact(expert_class):
    block = _FixedRoutes(expert_class).to("mps")
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


@pytest.mark.parametrize("broken", [False, True])
def test_real_expert_dispatch_selects_grouped_or_eager_from_probe(
    monkeypatch, expert_class, broken
):
    grouped_calls = []

    def safe_grouped(inputs, weights, offs):
        grouped_calls.append(offs.cpu().tolist())
        products = []
        start = 0
        for index, end in enumerate(offs.cpu().tolist()):
            products.append(inputs[start:end] @ weights[index])
            start = end
        return torch.cat(products)

    monkeypatch.setattr(hf_moe, "_grouped_mm", safe_grouped)
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: broken)
    registry = dict(hf_moe.ALL_EXPERTS_FUNCTIONS)
    block = _FixedRoutes(expert_class)
    inputs = torch.randn(10, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = _oracle(block, inputs)
    block.to("mps")
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = bridge(layer_input=inputs.to("mps"))
    assert block.experts.forward.__wrapped__ is type(block.experts).forward
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-6)
    assert len(caught) == int(broken)
    assert grouped_calls == ([] if broken else [[2, 5, 5, 10]] * 2)
    assert block.experts.config._experts_implementation == "grouped_mm"
    assert dict(hf_moe.ALL_EXPERTS_FUNCTIONS) == registry


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_autocast_cold_and_warm_dispatch_matches_raw_hf_policy(expert_class, weight_dtype):
    block = _FixedRoutes(expert_class).to("mps")
    block.experts.to(dtype=weight_dtype)
    reference = copy.deepcopy(block)
    bridge = MoEBridge("moe")
    bridge.set_original_component(block)
    inputs = torch.randn(10, 64, generator=torch.Generator().manual_seed(1)).to("mps")
    assert not moe._MPS_OUTPUT_SLICE_RESULTS
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.autocast("mps", dtype=torch.float16):
            first = bridge(inputs)
            assert torch.is_autocast_enabled("mps")
            key = (inputs.device, weight_dtype)
            assert set(moe._MPS_OUTPUT_SLICE_RESULTS) == {key}
            broken = moe._MPS_OUTPUT_SLICE_RESULTS[key]
            reference.experts.config._experts_implementation = "eager" if broken else "grouped_mm"
            expected = reference(inputs)
            second = bridge(inputs)
            assert torch.is_autocast_enabled("mps")
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert len(caught) == int(broken)
    assert set(moe._MPS_OUTPUT_SLICE_RESULTS) == {key}
    assert block.experts.config._experts_implementation == "grouped_mm"
