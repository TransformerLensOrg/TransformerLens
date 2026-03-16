import warnings
from unittest.mock import Mock, patch

import pytest
import torch

from transformer_lens.utilities.devices import (
    calculate_available_device_cuda_memory,
    determine_available_memory_for_available_devices,
    sort_devices_based_on_available_memory,
)
from transformer_lens.utils import get_device, warn_if_mps


def mock_available_devices(memory_stats: list[tuple[int, int]]):
    torch.cuda.device_count = Mock(return_value=len(memory_stats))

    def device_props_return(*args, **kwargs):
        total_memory = memory_stats[args[0]][0]
        device_props = Mock()
        device_props.total_memory = total_memory
        return device_props

    def memory_allocated_return(*args, **kwargs):
        return memory_stats[args[0]][1]

    torch.cuda.get_device_properties = Mock(side_effect=device_props_return)
    torch.cuda.memory_allocated = Mock(side_effect=memory_allocated_return)


def test_calculate_available_device_cuda_memory():
    mock_available_devices([(80, 40)])

    result = calculate_available_device_cuda_memory(0)
    assert result == 40


def test_determine_available_memory_for_available_devices():
    mock_available_devices(
        [
            (80, 60),
            (80, 15),
            (80, 40),
        ]
    )

    result = determine_available_memory_for_available_devices(3)

    assert result == [
        (0, 20),
        (1, 65),
        (2, 40),
    ]


def test_sort_devices_based_on_available_memory():
    devices = [
        (0, 20),
        (1, 65),
        (2, 40),
    ]

    result = sort_devices_based_on_available_memory(devices)

    assert result == [
        (1, 65),
        (2, 40),
        (0, 20),
    ]


# --- MPS warning / get_device tests ---


@pytest.fixture(autouse=True)
def reset_mps_warned():
    """Reset the _mps_warned flag before each test."""
    import transformer_lens.utils as utils_module

    utils_module._mps_warned = False
    yield
    utils_module._mps_warned = False


@patch.dict("os.environ", {}, clear=False)
@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.backends.mps.is_built", return_value=True)
def test_get_device_returns_cpu_when_mps_available(mock_built, mock_avail, mock_cuda):
    """get_device() should return CPU (not MPS) when MPS is available but env var is unset."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    result = get_device()
    assert result == torch.device("cpu")


@patch.dict("os.environ", {"TRANSFORMERLENS_ALLOW_MPS": "1"})
@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
@patch("torch.backends.mps.is_built", return_value=True)
def test_get_device_returns_mps_when_env_var_set(mock_built, mock_avail, mock_cuda):
    """get_device() should return MPS when TRANSFORMERLENS_ALLOW_MPS=1 is set."""
    result = get_device()
    assert result == torch.device("mps")


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_emits_warning():
    """warn_if_mps() should emit a UserWarning for MPS device."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_mps("mps")
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "MPS backend may produce silently incorrect results" in str(w[0].message)


@patch.dict("os.environ", {"TRANSFORMERLENS_ALLOW_MPS": "1"})
def test_warn_if_mps_silent_when_env_var_set():
    """warn_if_mps() should not warn when TRANSFORMERLENS_ALLOW_MPS=1 is set."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_mps("mps")
        assert len(w) == 0


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_silent_for_non_mps_device():
    """warn_if_mps() should not warn for cpu or cuda devices."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_mps("cpu")
        warn_if_mps("cuda")
        warn_if_mps(torch.device("cpu"))
        assert len(w) == 0


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_warns_only_once():
    """warn_if_mps() should only emit the warning once per process."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_mps("mps")
        warn_if_mps("mps")
        warn_if_mps(torch.device("mps"))
        assert len(w) == 1


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_handles_torch_device():
    """warn_if_mps() should handle torch.device objects correctly."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_mps(torch.device("mps"))
        assert len(w) == 1
        assert "MPS backend" in str(w[0].message)


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_suppressed_when_torch_version_safe():
    """warn_if_mps() should be silent when PyTorch meets the safe version threshold."""
    import os

    import transformer_lens.utils as utils_module

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    # Simulate a future safe version threshold below current torch
    original = utils_module._MPS_MIN_SAFE_TORCH_VERSION
    try:
        utils_module._MPS_MIN_SAFE_TORCH_VERSION = (1, 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
            assert len(w) == 0
    finally:
        utils_module._MPS_MIN_SAFE_TORCH_VERSION = original


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_active_when_torch_version_below_safe():
    """warn_if_mps() should warn when PyTorch is below the safe version threshold."""
    import os

    import transformer_lens.utils as utils_module

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    # Set threshold above any realistic current version
    original = utils_module._MPS_MIN_SAFE_TORCH_VERSION
    try:
        utils_module._MPS_MIN_SAFE_TORCH_VERSION = (99, 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
            assert len(w) == 1
    finally:
        utils_module._MPS_MIN_SAFE_TORCH_VERSION = original
