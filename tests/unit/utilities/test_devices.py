"""Tests for core device utilities."""

import warnings
from unittest.mock import Mock, patch

import pytest
import torch

from transformer_lens.utilities.devices import (
    ModelWithCfg,
    get_device,
    move_to_and_update_config,
    warn_if_mps,
)
from transformer_lens.utils import get_device, warn_if_mps


class MockModelWithCfg:
    """Mock model that implements the ModelWithCfg protocol."""

    def __init__(self, device="cpu", dtype=torch.float32):
        self.cfg = Mock()
        self.cfg.device = device
        self.cfg.dtype = dtype
        self._parameters = {"weight": torch.randn(10, 10)}

    def state_dict(self):
        return self._parameters

    def to(self, device_or_dtype):
        # Mock the to method
        if isinstance(device_or_dtype, torch.device):
            self.cfg.device = device_or_dtype.type
        elif isinstance(device_or_dtype, str):
            self.cfg.device = device_or_dtype
        elif isinstance(device_or_dtype, torch.dtype):
            self.cfg.dtype = device_or_dtype
        return self


def test_get_device_cuda_available():
    """Test get_device when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_device()
            assert device == torch.device("cuda")


@patch.dict("os.environ", {"TRANSFORMERLENS_ALLOW_MPS": "1"})
def test_get_device_mps_available():
    """Test get_device when MPS is available, PyTorch version >= 2.0, and env var set."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.backends.mps.is_built", return_value=True):
                with patch("torch.__version__", "2.0.0"):
                    device = get_device()
                    assert device == torch.device("mps")


def test_get_device_mps_pytorch_1x():
    """Test get_device when MPS is available but PyTorch version < 2.0."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("torch.backends.mps.is_built", return_value=True):
                with patch("torch.__version__", "1.13.0"):
                    device = get_device()
                    assert device == torch.device("cpu")


def test_get_device_cpu_fallback():
    """Test get_device falls back to CPU when no GPU available."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=False):
            device = get_device()
            assert device == torch.device("cpu")


def test_model_with_cfg_protocol():
    """Test that ModelWithCfg protocol is runtime checkable."""
    model = MockModelWithCfg()
    assert isinstance(model, ModelWithCfg)

    # Test that it has required attributes
    assert hasattr(model, "cfg")
    assert hasattr(model, "state_dict")
    assert hasattr(model, "to")


def test_move_to_and_update_config_device():
    """Test move_to_and_update_config with device."""
    model = MockModelWithCfg(device="cpu")

    with patch("torch.nn.Module.to") as mock_to:
        mock_to.return_value = model
        result = move_to_and_update_config(model, torch.device("cuda"))

        assert model.cfg.device == "cuda"
        assert result == model
        mock_to.assert_called_once_with(model, torch.device("cuda"))


def test_move_to_and_update_config_string():
    """Test move_to_and_update_config with string device."""
    model = MockModelWithCfg(device="cpu")

    with patch("torch.nn.Module.to") as mock_to:
        mock_to.return_value = model
        result = move_to_and_update_config(model, "mps")

        assert model.cfg.device == "mps"
        assert result == model
        mock_to.assert_called_once_with(model, "mps")


def test_move_to_and_update_config_dtype():
    """Test move_to_and_update_config with dtype."""
    model = MockModelWithCfg(dtype=torch.float32)

    with patch("torch.nn.Module.to") as mock_to:
        mock_to.return_value = model
        result = move_to_and_update_config(model, torch.float16)

        assert model.cfg.dtype == torch.float16
        assert result == model
        mock_to.assert_called_once_with(model, torch.float16)


def test_move_to_and_update_config_dtype_no_dtype_attr():
    """Test move_to_and_update_config with dtype when model has no dtype attr."""
    model = MockModelWithCfg()
    delattr(model.cfg, "dtype")  # Remove dtype attribute

    with patch("torch.nn.Module.to") as mock_to:
        mock_to.return_value = model
        result = move_to_and_update_config(model, torch.float16)

        # Should not crash and should still call the base to method
        assert result == model
        mock_to.assert_called_once_with(model, torch.float16)


def test_move_to_and_update_config_print_details_false():
    """Test move_to_and_update_config with print_details=False."""
    model = MockModelWithCfg(device="cpu")

    with patch("torch.nn.Module.to") as mock_to:
        mock_to.return_value = model
        with patch("builtins.print") as mock_print:
            result = move_to_and_update_config(model, "mps", print_details=False)

            assert model.cfg.device == "mps"
            assert result == model
            mock_print.assert_not_called()


# --- MPS warning / get_device tests ---


@pytest.fixture(autouse=True)
def reset_mps_warned():
    """Reset the _mps_warned and _mps_broken_torch_warned flags before each test."""
    import transformer_lens.utilities.devices as devices_module

    devices_module._mps_warned = False
    devices_module._mps_broken_torch_warned = False
    yield
    devices_module._mps_warned = False
    devices_module._mps_broken_torch_warned = False


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

    import transformer_lens.utilities.devices as devices_module

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    # Simulate a future safe version threshold below current torch
    original = devices_module._MPS_MIN_SAFE_TORCH_VERSION
    try:
        devices_module._MPS_MIN_SAFE_TORCH_VERSION = (1, 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
            assert len(w) == 0
    finally:
        devices_module._MPS_MIN_SAFE_TORCH_VERSION = original


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_active_when_torch_version_below_safe():
    """warn_if_mps() should warn when PyTorch is below the safe version threshold."""
    import os

    import transformer_lens.utilities.devices as devices_module

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    # Set threshold above any realistic current version
    original = devices_module._MPS_MIN_SAFE_TORCH_VERSION
    try:
        devices_module._MPS_MIN_SAFE_TORCH_VERSION = (99, 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
            assert len(w) == 1
    finally:
        devices_module._MPS_MIN_SAFE_TORCH_VERSION = original


# --- Known-broken-torch-on-MPS warning tests (issue #1062, torch 2.8.0) ---


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_warns_about_broken_torch_version():
    """When torch is in _MPS_BROKEN_TORCH_VERSIONS, warn_if_mps emits the broken-version warning."""
    import os

    import transformer_lens.utilities.devices as devices_module

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    with patch(
        "transformer_lens.utilities.devices._torch_version_tuple",
        return_value=(2, 8),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
        messages = [str(warning.message) for warning in w]
        assert any(
            "known MPS bug that produces silently incorrect results" in m for m in messages
        ), f"Expected broken-torch warning in {messages}"
        assert any("issues/1062" in m for m in messages)


@patch.dict("os.environ", {"TRANSFORMERLENS_ALLOW_MPS": "1"})
def test_warn_if_mps_broken_torch_warning_fires_even_when_opted_in():
    """The broken-torch warning must fire even with TRANSFORMERLENS_ALLOW_MPS=1,
    because the bug produces silently wrong output regardless of opt-in."""
    with patch(
        "transformer_lens.utilities.devices._torch_version_tuple",
        return_value=(2, 8),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
        messages = [str(warning.message) for warning in w]
        assert any("known MPS bug" in m for m in messages)


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_no_broken_warning_on_safe_torch_version():
    """Non-broken torch versions should not emit the broken-torch warning."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    for version in [(2, 7), (2, 9), (3, 0)]:
        with patch(
            "transformer_lens.utilities.devices._torch_version_tuple",
            return_value=version,
        ):
            # Reset the broken-warn flag for each iteration
            import transformer_lens.utilities.devices as devices_module

            devices_module._mps_broken_torch_warned = False
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_if_mps("mps")
            messages = [str(warning.message) for warning in w]
            assert not any(
                "known MPS bug" in m for m in messages
            ), f"Unexpected broken-torch warning on torch {version}: {messages}"


@patch.dict("os.environ", {}, clear=False)
def test_warn_if_mps_broken_warning_fires_only_once():
    """The broken-torch warning should only fire once per process."""
    import os

    os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
    with patch(
        "transformer_lens.utilities.devices._torch_version_tuple",
        return_value=(2, 8),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
            warn_if_mps("mps")
            warn_if_mps(torch.device("mps"))
        broken_warnings = [warning for warning in w if "known MPS bug" in str(warning.message)]
        assert len(broken_warnings) == 1


def test_torch_mps_has_known_broken_bug_for_2_8():
    """_torch_mps_has_known_broken_bug should return True for torch 2.8."""
    from transformer_lens.utilities.devices import _torch_mps_has_known_broken_bug

    with patch(
        "transformer_lens.utilities.devices._torch_version_tuple",
        return_value=(2, 8),
    ):
        assert _torch_mps_has_known_broken_bug() is True


def test_torch_mps_has_known_broken_bug_false_for_other_versions():
    """_torch_mps_has_known_broken_bug should return False for non-broken torch versions."""
    from transformer_lens.utilities.devices import _torch_mps_has_known_broken_bug

    for version in [(2, 7), (2, 9), (3, 0)]:
        with patch(
            "transformer_lens.utilities.devices._torch_version_tuple",
            return_value=version,
        ):
            assert (
                _torch_mps_has_known_broken_bug() is False
            ), f"torch {version} incorrectly flagged as broken"
