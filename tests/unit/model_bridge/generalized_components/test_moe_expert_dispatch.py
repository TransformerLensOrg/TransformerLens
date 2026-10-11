"""Scoped Granite expert dispatch without shared configuration mutation."""

import copy
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridExperts,
)

from transformer_lens.model_bridge.generalized_components.moe import (
    _granite_mps_experts_forward,
    _guard_granite_mps_experts,
)


class _MpsInput(torch.Tensor):
    @property
    def device(self):
        return torch.device("mps")


class _Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fail = False

    def eager(self, hidden_states):
        if self.fail:
            raise ValueError("expert failure")
        return "eager"

    @wraps(eager)
    def forward(self, hidden_states):
        return self.config._experts_implementation


def test_dispatch_does_not_mutate_shared_config():
    config = SimpleNamespace(_experts_implementation="grouped_mm")
    experts = [_Experts(config), _Experts(config)]
    mps = torch.empty(1).as_subclass(_MpsInput)
    cpu = torch.empty(1)
    with pytest.warns(RuntimeWarning, match="eager Granite"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda pair: _granite_mps_experts_forward(*pair),
                    [(experts[0], mps), (experts[1], cpu)],
                )
            )
    assert results == ["eager", "grouped_mm"]
    assert vars(config) == {"_experts_implementation": "grouped_mm"}


@pytest.mark.parametrize("policy", ["eager", "batched_mm", None])
def test_other_policies_are_unchanged(policy):
    experts = _Experts(SimpleNamespace(_experts_implementation=policy))
    assert _granite_mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput)) == policy


def test_exception_does_not_leave_dispatch_state():
    experts = _Experts(SimpleNamespace(_experts_implementation="grouped_mm"))
    experts.fail = True
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="expert failure"):
        _granite_mps_experts_forward(experts, hidden_states=torch.empty(1).as_subclass(_MpsInput))
    assert experts.config._experts_implementation == "grouped_mm"
    experts.fail = False
    with pytest.warns(RuntimeWarning):
        assert (
            _granite_mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput)) == "eager"
        )


def _block():
    config = SimpleNamespace(
        hidden_size=8, intermediate_size=4, num_local_experts=4, hidden_act="silu"
    )
    config._experts_implementation = "grouped_mm"
    block = nn.Module()
    block.experts = GraniteMoeHybridExperts(config)
    return block


@pytest.mark.parametrize(
    "torch_version,hf_version,installed",
    [
        ("2.11.0", "5.13.0", True),
        ("2.11.0+local", "5.13.0+local", True),
        ("2.11.1", "5.13.0", False),
        ("2.12.0", "5.13.0", False),
        ("2.11.0", "5.14.0", False),
        ("2.11.0rc1", "5.13.0", False),
        ("2.11.0", "5.13.0.dev0", False),
    ],
)
def test_installation_version_scope(monkeypatch, torch_version, hf_version, installed):
    import transformers

    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(transformers, "__version__", hf_version)
    block = _block()
    _guard_granite_mps_experts(block)
    assert (block.experts.forward.__func__ is _granite_mps_experts_forward) == installed
    assert block.experts.config._experts_implementation == "grouped_mm"


def test_installation_is_idempotent_and_deepcopy_safe(monkeypatch):
    import transformers

    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    block = _block()
    _guard_granite_mps_experts(block)
    forward = block.experts.forward
    _guard_granite_mps_experts(block)
    assert block.experts.forward is forward
    cloned = copy.deepcopy(block)
    assert cloned.experts.forward.__self__ is cloned.experts


def test_guarded_cpu_experts_keep_grouped_execution_without_warning(monkeypatch):
    import transformers

    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    block = _block()
    with torch.no_grad():
        for parameter in block.experts.parameters():
            parameter.normal_(std=0.1)
    inputs = torch.randn(5, 8)
    indices = torch.tensor([[0], [0], [1], [3], [3]])
    weights = torch.ones(5, 1)
    with torch.no_grad():
        expected = block.experts(inputs, indices, weights)
    _guard_granite_mps_experts(block)
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        actual = block.experts(inputs, indices, weights)
    assert not caught
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert block.experts.config._experts_implementation == "grouped_mm"


def test_unrelated_experts_and_custom_forwards_are_unchanged():
    block = nn.Module()
    block.experts = _Experts(SimpleNamespace(_experts_implementation="grouped_mm"))
    _guard_granite_mps_experts(block)
    assert block.experts.forward.__func__ is _Experts.forward
    block = _block()
    custom = lambda hidden_states: hidden_states
    block.experts.forward = custom
    _guard_granite_mps_experts(block)
    assert block.experts.forward is custom
