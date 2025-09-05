"""Tests for core device utilities."""

from unittest.mock import Mock, patch

import torch

from transformer_lens.utilities.devices import (
    ModelWithCfg,
    get_device,
    move_to_and_update_config,
)


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


def test_get_device_mps_available():
    """Test get_device when MPS is available and PyTorch version >= 2.0."""
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
