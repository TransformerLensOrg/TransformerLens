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

from transformer_lens.model_bridge.generalized_components import moe
from transformer_lens.model_bridge.generalized_components.moe import (
    _granite_mps_experts_forward,
    _guard_granite_mps_experts,
    _mps_output_slice_writes_are_broken,
    _probe_mps_output_slice_writes,
)


@pytest.fixture(autouse=True)
def isolated_probe(monkeypatch):
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: True)
    yield
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()


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
    "torch_version,hf_version",
    [
        ("2.11.0", "5.13.0"),
        ("2.11.0+local", "5.13.0+local"),
        ("2.11.1", "5.13.0"),
        ("2.12.0", "5.13.0"),
        ("2.11.0", "5.14.0"),
        ("2.11.0rc1", "5.13.0"),
        ("2.11.0", "5.13.0.dev0"),
    ],
)
@pytest.mark.parametrize("broken", [False, True])
def test_dispatch_uses_probe_not_package_versions(monkeypatch, torch_version, hf_version, broken):
    import transformers

    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(transformers, "__version__", hf_version)
    probes = []

    def probe(device, dtype):
        probes.append((device, dtype))
        return broken

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    block = _block()
    _guard_granite_mps_experts(block)
    assert block.experts.forward.__func__ is _granite_mps_experts_forward
    assert not probes
    assert block.experts.config._experts_implementation == "grouped_mm"
    experts = _Experts(block.experts.config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _granite_mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput))
    assert actual == ("eager" if broken else "grouped_mm")
    assert len(caught) == int(broken)
    assert probes == [(torch.device("mps"), torch.float32)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("corruption", [None, "missing_write", "prefix_write"])
def test_primitive_probe_checks_products_and_untouched_rows(monkeypatch, dtype, corruption):
    original_mm = torch.mm

    def mm(inputs, weights, *, out):
        assert not torch.is_grad_enabled()
        assert out.storage_offset() == 5 * 32
        if corruption != "missing_write":
            original_mm(inputs, weights, out=out)
        if corruption == "prefix_write":
            out._base[0, 0] = 1
        return out

    monkeypatch.setattr(torch, "mm", mm)
    rng_before = torch.get_rng_state().clone()
    assert _probe_mps_output_slice_writes(torch.device("cpu"), dtype) == (corruption is not None)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


def test_probe_cache_is_per_device_and_dtype(monkeypatch):
    calls = []

    def probe(device, dtype):
        calls.append((device, dtype))
        return dtype == torch.float16

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    keys = [
        (torch.device("mps:0"), torch.float32),
        (torch.device("mps:0"), torch.float16),
        (torch.device("mps:1"), torch.float32),
    ]
    for _ in range(2):
        for device, dtype in keys:
            assert _mps_output_slice_writes_are_broken(device, dtype) == (dtype == torch.float16)
    assert calls == keys


def test_concurrent_first_use_probes_once(monkeypatch):
    calls = []

    def probe(device, dtype):
        calls.append((device, dtype))
        return False

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: _mps_output_slice_writes_are_broken(torch.device("mps"), torch.float32),
                range(16),
            )
        )
    assert results == [False] * 16
    assert calls == [(torch.device("mps"), torch.float32)]


@pytest.mark.parametrize("error_type", [RuntimeError, NotImplementedError, TypeError])
def test_probe_failure_is_diagnostic_and_not_cached(monkeypatch, error_type):
    error = error_type("unsupported diagnostic")

    def probe(device, dtype):
        raise error

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    experts = _Experts(SimpleNamespace(_experts_implementation="grouped_mm"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="Select eager experts explicitly") as raised:
            _granite_mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput))
    assert raised.value.__cause__ is error
    assert not caught
    assert not moe._MPS_OUTPUT_SLICE_RESULTS
    assert experts.config._experts_implementation == "grouped_mm"
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: False)
    assert (
        _granite_mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput)) == "grouped_mm"
    )


def test_unexpected_probe_failure_propagates(monkeypatch):
    def probe(device, dtype):
        raise ValueError("unexpected diagnostic failure")

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    with pytest.raises(ValueError, match="unexpected diagnostic failure"):
        _mps_output_slice_writes_are_broken(torch.device("mps"), torch.float32)
    assert not moe._MPS_OUTPUT_SLICE_RESULTS


@pytest.mark.parametrize(
    "device,policy",
    [
        (device, policy)
        for device in ["cpu", "cuda"]
        for policy in ["grouped_mm", "eager", "batched_mm", None]
    ]
    + [("mps", policy) for policy in ["eager", "batched_mm", None]],
)
def test_unaffected_calls_do_not_probe(monkeypatch, policy, device):
    class DeviceInput(torch.Tensor):
        @property
        def device(self):
            return torch.device(device)

    def probe(device, dtype):
        raise AssertionError("unaffected execution must not probe")

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    experts = _Experts(SimpleNamespace(_experts_implementation=policy))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            _granite_mps_experts_forward(experts, torch.empty(1).as_subclass(DeviceInput)) == policy
        )
    assert not caught
    assert not moe._MPS_OUTPUT_SLICE_RESULTS


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


def test_dispatch_does_not_create_missing_policy():
    class MissingPolicyExperts(_Experts):
        def forward(self, hidden_states):
            return hidden_states

    experts = MissingPolicyExperts(SimpleNamespace())
    inputs = torch.empty(1).as_subclass(_MpsInput)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _granite_mps_experts_forward(experts, inputs) is inputs
    assert not caught
    assert vars(experts.config) == {}


def test_guarded_eager_cpu_forward_preserves_hooks_and_gradients(monkeypatch):
    import transformers

    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    block = _block()
    block.experts.config._experts_implementation = "eager"
    with torch.no_grad():
        for parameter in block.experts.parameters():
            parameter.normal_(std=0.1)
    reference = copy.deepcopy(block)
    _guard_granite_mps_experts(block)
    inputs = torch.randn(5, 8, requires_grad=True)
    routes = torch.rand(5, 1, requires_grad=True)
    indices = torch.tensor([[0], [0], [1], [3], [3]])
    calls = []
    handle = block.experts.register_forward_hook(lambda module, args, output: calls.append(args))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = block.experts(inputs, indices, routes)
        expected = reference.experts(inputs, indices, routes)
        assert not caught
        assert len(calls) == 1
        assert calls[0][1] is indices
        assert calls[0][2] is routes
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(
            actual.sum(), (inputs, routes, *block.experts.parameters())
        )
        expected_grads = torch.autograd.grad(
            expected.sum(), (inputs, routes, *reference.experts.parameters())
        )
        for measured, oracle in zip(actual_grads, expected_grads):
            torch.testing.assert_close(measured, oracle, rtol=0, atol=0)
        assert block.experts.config._experts_implementation == "eager"
    finally:
        handle.remove()


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
