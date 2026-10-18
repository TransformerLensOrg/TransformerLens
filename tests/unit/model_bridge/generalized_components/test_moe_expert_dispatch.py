"""Shared HF expert dispatch without shared configuration mutation."""

import copy
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from inspect import signature
from threading import Lock, Thread
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.integrations import moe as hf_moe
from transformers.integrations.moe import ExpertsInterface, use_experts_implementation
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeExperts
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridExperts,
)
from transformers.models.mixtral.modeling_mixtral import MixtralExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from transformer_lens.model_bridge.generalized_components import moe
from transformer_lens.model_bridge.generalized_components.moe import (
    _guard_mps_experts,
    _hf_eager_experts_forward,
    _mps_experts_forward,
    _mps_output_slice_writes_are_broken,
    _probe_mps_output_slice_writes,
)

_EXPERT_CLASSES = [GraniteMoeHybridExperts, GraniteMoeExperts, MixtralExperts, Qwen3MoeExperts]


@pytest.fixture(autouse=True)
def isolated_probe(monkeypatch):
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()
    moe._MPS_EXPERTS_WARNED = False
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: True)
    yield
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()
    moe._MPS_EXPERTS_WARNED = False


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


@pytest.fixture
def stub_dispatch(monkeypatch):
    recognize = moe._hf_eager_experts_forward

    def eager(forward):
        if forward is _Experts.forward:
            return forward.__wrapped__
        return recognize(forward)

    monkeypatch.setattr(moe, "_hf_eager_experts_forward", eager)


def test_dispatch_does_not_mutate_shared_config(stub_dispatch):
    config = SimpleNamespace(_experts_implementation="grouped_mm")
    experts = [_Experts(config), _Experts(config)]
    mps = torch.empty(1).as_subclass(_MpsInput)
    cpu = torch.empty(1)
    with pytest.warns(RuntimeWarning, match="eager experts"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda pair: _mps_experts_forward(*pair),
                    [(experts[0], mps), (experts[1], cpu)],
                )
            )
    assert results == ["eager", "grouped_mm"]
    assert vars(config) == {"_experts_implementation": "grouped_mm"}


@pytest.mark.parametrize("policy", ["eager", "batched_mm", None])
def test_other_policies_are_unchanged(policy):
    experts = _Experts(SimpleNamespace(_experts_implementation=policy))
    assert _mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput)) == policy


def test_exception_does_not_leave_dispatch_state(stub_dispatch):
    experts = _Experts(SimpleNamespace(_experts_implementation="grouped_mm"))
    experts.fail = True
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="expert failure"):
        _mps_experts_forward(experts, hidden_states=torch.empty(1).as_subclass(_MpsInput))
    assert experts.config._experts_implementation == "grouped_mm"
    experts.fail = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput)) == "eager"
    assert not caught


def _block(expert_class=GraniteMoeHybridExperts):
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=4,
        moe_intermediate_size=4,
        num_local_experts=4,
        num_experts=4,
        hidden_act="silu",
    )
    config._experts_implementation = "grouped_mm"
    block = nn.Module()
    block.experts = expert_class(config)
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
def test_dispatch_uses_probe_not_package_versions(
    monkeypatch, torch_version, hf_version, broken, stub_dispatch
):
    import transformers

    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(transformers, "__version__", hf_version)
    probes = []

    def probe(device, dtype):
        probes.append((device, dtype))
        return broken

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    block = _block()
    _guard_mps_experts(block)
    assert block.experts.forward.__wrapped__ is type(block.experts).forward
    assert not probes
    assert block.experts.config._experts_implementation == "grouped_mm"
    experts = _Experts(block.experts.config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput))
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
def test_probe_failure_is_diagnostic_and_not_cached(monkeypatch, error_type, stub_dispatch):
    error = error_type("unsupported diagnostic")

    def probe(device, dtype):
        raise error

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    experts = _Experts(SimpleNamespace(_experts_implementation="grouped_mm"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="Select eager experts explicitly") as raised:
            _mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput))
    assert raised.value.__cause__ is error
    assert not caught
    assert not moe._MPS_OUTPUT_SLICE_RESULTS
    assert experts.config._experts_implementation == "grouped_mm"
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: False)
    assert _mps_experts_forward(experts, torch.empty(1).as_subclass(_MpsInput)) == "grouped_mm"


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
        assert _mps_experts_forward(experts, torch.empty(1).as_subclass(DeviceInput)) == policy
    assert not caught
    assert not moe._MPS_OUTPUT_SLICE_RESULTS


@pytest.mark.parametrize("expert_class", _EXPERT_CLASSES)
def test_installation_is_idempotent_and_deepcopy_safe(monkeypatch, expert_class):
    import transformers

    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    block = _block(expert_class)
    original_signature = signature(block.experts.forward)
    _guard_mps_experts(block)
    forward = block.experts.forward
    assert signature(forward) == original_signature
    _guard_mps_experts(block)
    assert block.experts.forward is forward
    cloned = copy.deepcopy(block)
    assert cloned.experts.forward.__self__ is cloned.experts
    assert signature(cloned.experts.forward) == original_signature


@pytest.mark.parametrize("expert_class", _EXPERT_CLASSES)
def test_guarded_cpu_experts_keep_grouped_execution_without_warning(monkeypatch, expert_class):
    import transformers

    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    block = _block(expert_class)
    with torch.no_grad():
        for parameter in block.experts.parameters():
            parameter.normal_(std=0.1)
    inputs = torch.randn(5, 8)
    indices = torch.tensor([[0], [0], [1], [3], [3]])
    weights = torch.ones(5, 1)
    with torch.no_grad():
        expected = block.experts(inputs, indices, weights)
    _guard_mps_experts(block)
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
        assert _mps_experts_forward(experts, inputs) is inputs
    assert not caught
    assert vars(experts.config) == {}


@pytest.mark.parametrize("expert_class", _EXPERT_CLASSES)
def test_guarded_eager_cpu_forward_preserves_hooks_and_gradients(monkeypatch, expert_class):
    import transformers

    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(transformers, "__version__", "5.13.0")
    block = _block(expert_class)
    block.experts.config._experts_implementation = "eager"
    with torch.no_grad():
        for parameter in block.experts.parameters():
            parameter.normal_(std=0.1)
    reference = copy.deepcopy(block)
    _guard_mps_experts(block)
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
    _guard_mps_experts(block)
    assert block.experts.forward.__func__ is _Experts.forward
    block = _block()
    custom = lambda hidden_states: hidden_states
    block.experts.forward = custom
    _guard_mps_experts(block)
    assert block.experts.forward is custom


def _decorated_block(interface, config=None):
    @use_experts_implementation(experts_interface=interface)
    class DecoratedExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.received = None

        def forward(self, hidden_states, token=None, *, payload=None):
            self.received = (hidden_states, token, payload)
            return "eager"

    block = nn.Module()
    block.experts = DecoratedExperts(
        config if config is not None else SimpleNamespace(_experts_implementation="grouped_mm")
    )
    return block


@pytest.mark.parametrize("broken", [False, True])
@pytest.mark.parametrize("keyword_input", [False, True])
def test_shared_decorator_dispatch_preserves_arguments_and_global_registry(
    monkeypatch, broken, keyword_input
):
    def grouped(experts, hidden_states, token=None, *, payload=None):
        experts.received = (hidden_states, token, payload)
        return "grouped_mm"

    registry = dict(hf_moe.ALL_EXPERTS_FUNCTIONS)
    interface = ExpertsInterface()
    interface["grouped_mm"] = grouped
    monkeypatch.setattr(hf_moe, "grouped_mm_experts_forward", grouped)
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: broken)
    block = _decorated_block(interface)
    _guard_mps_experts(block)
    assert block.experts.forward.__wrapped__ is type(block.experts).forward
    inputs = torch.empty(1).as_subclass(_MpsInput)
    token, payload = object(), object()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if keyword_input:
            actual = block.experts(hidden_states=inputs, token=token, payload=payload)
        else:
            actual = block.experts(inputs, token, payload=payload)
    assert actual == ("eager" if broken else "grouped_mm")
    assert block.experts.received == (inputs, token, payload)
    assert len(caught) == int(broken)
    assert vars(block.experts.config) == {"_experts_implementation": "grouped_mm"}
    assert dict(hf_moe.ALL_EXPERTS_FUNCTIONS) == registry


def test_guarded_shared_config_supports_concurrent_cpu_and_mps_dispatch(monkeypatch):
    def grouped(experts, hidden_states):
        return "grouped_mm"

    interface = ExpertsInterface()
    interface["grouped_mm"] = grouped
    monkeypatch.setattr(hf_moe, "grouped_mm_experts_forward", grouped)
    config = SimpleNamespace(_experts_implementation="grouped_mm")
    blocks = [_decorated_block(interface, config), _decorated_block(interface, config)]
    for block in blocks:
        _guard_mps_experts(block)
    inputs = [torch.empty(1).as_subclass(_MpsInput), torch.empty(1)]
    with pytest.warns(RuntimeWarning, match="eager experts"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(lambda pair: pair[0].experts(pair[1]), zip(blocks, inputs)))
    assert actual == ["eager", "grouped_mm"]
    assert all(block.experts.config is config for block in blocks)
    assert vars(config) == {"_experts_implementation": "grouped_mm"}


@pytest.mark.parametrize("expert_class", _EXPERT_CLASSES)
def test_recovers_real_expert_eager_forward(expert_class):
    assert _hf_eager_experts_forward(expert_class.forward) is expert_class.forward.__wrapped__


def test_custom_grouped_interface_is_not_overridden():
    def custom(experts, hidden_states):
        return "custom"

    interface = ExpertsInterface()
    interface["grouped_mm"] = custom
    block = _decorated_block(interface)
    forward = block.experts.forward
    _guard_mps_experts(block)
    assert block.experts.forward == forward
    with warnings.catch_warnings(record=True) as caught:
        assert block.experts(torch.empty(1).as_subclass(_MpsInput)) == "custom"
    assert not caught
    assert not moe._MPS_OUTPUT_SLICE_RESULTS


def test_interface_changed_after_installation_is_respected():
    interface = ExpertsInterface()
    block = _decorated_block(interface)
    _guard_mps_experts(block)
    assert block.experts.forward.__wrapped__ is type(block.experts).forward
    interface["grouped_mm"] = lambda experts, hidden_states: "custom"
    with warnings.catch_warnings(record=True) as caught:
        assert block.experts(torch.empty(1).as_subclass(_MpsInput)) == "custom"
    assert not caught
    assert not moe._MPS_OUTPUT_SLICE_RESULTS


def test_mismatched_wrapped_forward_is_not_recognized(monkeypatch):
    block = _decorated_block(ExpertsInterface())
    forward = type(block.experts).forward
    monkeypatch.setattr(forward, "__wrapped__", lambda experts, inputs: inputs)
    _guard_mps_experts(block)
    assert block.experts.forward.__func__ is forward


def test_ordinary_expert_lists_are_unchanged():
    block = nn.Module()
    block.experts = nn.ModuleList([nn.Linear(8, 4)])
    forward = block.experts.forward
    _guard_mps_experts(block)
    assert block.experts.forward == forward


@pytest.mark.parametrize("concurrent", [False, True])
def test_warning_is_once_across_layers_models_and_forwards(concurrent):
    blocks = [_decorated_block(ExpertsInterface()) for _ in range(4)]
    for block in blocks:
        _guard_mps_experts(block)
    inputs = torch.empty(1).as_subclass(_MpsInput)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if concurrent:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda block: block.experts(inputs), blocks * 8))
        else:
            results = [block.experts(inputs) for block in blocks * 8]
    assert results == ["eager"] * 32
    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning
    assert "eager experts" in str(caught[0].message)
    assert moe._MPS_EXPERTS_WARNED


def test_warning_identifies_external_caller_through_runtime_wrappers(monkeypatch):
    modules = []

    def currentframe():
        frame = sys._getframe(1)
        while frame is not None:
            modules.append(frame.f_globals.get("__name__", ""))
            frame = frame.f_back
        return sys._getframe(1)

    monkeypatch.setattr(moe, "currentframe", currentframe)
    block = _decorated_block(ExpertsInterface())
    _guard_mps_experts(block)
    assert not moe._MPS_EXPERTS_WARNED
    inputs = torch.empty(1).as_subclass(_MpsInput)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        caller_line = sys._getframe().f_lineno + 1
        result = block.experts(inputs)
    assert result == "eager"
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == caller_line
    assert any(module.startswith("jaxtyping.") for module in modules)
    assert any(module.startswith("torch.") for module in modules)


def test_warning_as_error_propagates_without_consuming_notification():
    block = _decorated_block(ExpertsInterface())
    _guard_mps_experts(block)
    inputs = torch.empty(1).as_subclass(_MpsInput)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(2):
            with pytest.raises(RuntimeWarning, match="eager experts"):
                block.experts(inputs)
            assert not moe._MPS_EXPERTS_WARNED
            assert block.experts.received is None
            assert block.experts.config._experts_implementation == "grouped_mm"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert block.experts(inputs) == "eager"
        assert block.experts(inputs) == "eager"
    assert len(caught) == 1
    assert moe._MPS_EXPERTS_WARNED


def test_unaffected_calls_do_not_consume_fallback_notification(monkeypatch):
    def grouped(experts, hidden_states):
        return "grouped_mm"

    interface = ExpertsInterface()
    interface["grouped_mm"] = grouped
    monkeypatch.setattr(hf_moe, "grouped_mm_experts_forward", grouped)
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: False)
    block = _decorated_block(interface)
    _guard_mps_experts(block)
    inputs = torch.empty(1).as_subclass(_MpsInput)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert block.experts(torch.empty(1)) == "grouped_mm"
        assert block.experts(inputs) == "grouped_mm"
        block.experts.config._experts_implementation = "eager"
        assert block.experts(inputs) == "eager"
    assert not caught
    assert not moe._MPS_EXPERTS_WARNED
    block.experts.config._experts_implementation = "grouped_mm"
    moe._MPS_OUTPUT_SLICE_RESULTS.clear()
    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", lambda device, dtype: True)
    with pytest.warns(RuntimeWarning, match="eager experts"):
        assert block.experts(inputs) == "eager"


def test_fallback_warning_does_not_suppress_unrelated_warnings():
    block = _decorated_block(ExpertsInterface())
    _guard_mps_experts(block)
    inputs = torch.empty(1).as_subclass(_MpsInput)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert block.experts(inputs) == "eager"
        warnings.warn("Unrelated diagnostic", UserWarning)
        assert block.experts(inputs) == "eager"
        warnings.warn("Unrelated diagnostic", UserWarning)
    assert [warning.category for warning in caught] == [RuntimeWarning, UserWarning, UserWarning]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("broken", [False, True])
def test_primitive_probe_is_independent_of_ambient_autocast(monkeypatch, dtype, broken):
    original_mm = torch.mm

    def mm(inputs, weights, *, out):
        assert not torch.is_autocast_enabled("cpu")
        assert inputs.dtype == weights.dtype == out.dtype == dtype
        if broken:
            return out
        return original_mm(inputs, weights, out=out)

    monkeypatch.setattr(torch, "mm", mm)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _probe_mps_output_slice_writes(torch.device("cpu"), dtype) == broken
        assert torch.is_autocast_enabled("cpu")


def test_dispatch_probes_batched_weight_dtype_instead_of_input_dtype(monkeypatch):
    block = _decorated_block(ExpertsInterface())
    block.experts.register_parameter(
        "gate_up_proj", nn.Parameter(torch.empty(4, 8, 8, dtype=torch.float16))
    )
    block.experts.register_parameter(
        "down_proj", nn.Parameter(torch.empty(4, 8, 4, dtype=torch.float16))
    )
    probed = []

    def probe(device, dtype):
        probed.append(dtype)
        return True

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    _guard_mps_experts(block)
    with pytest.warns(RuntimeWarning, match="eager experts"):
        assert block.experts(torch.empty(1).as_subclass(_MpsInput)) == "eager"
    assert probed == [torch.float16]


def test_warning_callback_can_reenter_guarded_experts(monkeypatch):
    monkeypatch.setattr(moe, "_MPS_EXPERTS_WARNING_LOCK", Lock())
    block = _decorated_block(ExpertsInterface())
    _guard_mps_experts(block)
    inputs = torch.empty(1).as_subclass(_MpsInput)
    callbacks, results, errors = [], [], []

    def showwarning(*args, **kwargs):
        callbacks.append(block.experts(inputs))

    def invoke():
        try:
            results.append(block.experts(inputs))
        except Exception as error:
            errors.append(error)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", showwarning)
        worker = Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout=2)
    assert not worker.is_alive(), "warning callback deadlocked on re-entry"
    assert not errors
    assert callbacks == ["eager"]
    assert results == ["eager"]
    assert moe._MPS_EXPERTS_WARNED


def test_warning_callback_failure_releases_reservation(monkeypatch):
    block = _decorated_block(ExpertsInterface())
    _guard_mps_experts(block)
    inputs = torch.empty(1).as_subclass(_MpsInput)

    def showwarning(*args, **kwargs):
        raise ValueError("warning callback failed")

    with warnings.catch_warnings(), monkeypatch.context() as context:
        warnings.simplefilter("always")
        context.setattr(warnings, "showwarning", showwarning)
        with pytest.raises(ValueError, match="warning callback failed"):
            block.experts(inputs)
    assert not moe._MPS_EXPERTS_WARNED
    with pytest.warns(RuntimeWarning, match="eager experts"):
        assert block.experts(inputs) == "eager"


def test_dispatch_checks_both_projection_weight_dtypes(monkeypatch):
    def grouped(experts, hidden_states):
        return "grouped_mm"

    interface = ExpertsInterface()
    interface["grouped_mm"] = grouped
    monkeypatch.setattr(hf_moe, "grouped_mm_experts_forward", grouped)
    block = _decorated_block(interface)
    block.experts.register_parameter(
        "gate_up_proj", nn.Parameter(torch.empty(4, 8, 8, dtype=torch.float16))
    )
    block.experts.register_parameter(
        "down_proj", nn.Parameter(torch.empty(4, 8, 4, dtype=torch.float32))
    )
    probed = []

    def probe(device, dtype):
        probed.append(dtype)
        return False

    monkeypatch.setattr(moe, "_probe_mps_output_slice_writes", probe)
    _guard_mps_experts(block)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert block.experts(torch.empty(1).as_subclass(_MpsInput)) == "grouped_mm"
    assert set(probed) == {torch.float16, torch.float32}
    assert not caught
