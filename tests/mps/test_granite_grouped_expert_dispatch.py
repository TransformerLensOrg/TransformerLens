"""Granite expert correctness on MPS with fixed routes and an independent oracle."""

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

    def forward(self, layer_input):
        return self.experts(layer_input, self.indices, self.weights)


def _oracle(block, inputs):
    contributions = []
    for token, indices, weights in zip(inputs, block.indices, block.weights):
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
