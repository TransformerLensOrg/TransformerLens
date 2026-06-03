"""Tests for multi-GPU utilities."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from transformer_lens.utilities import (
    calculate_available_device_cuda_memory,
    determine_available_memory_for_available_devices,
    sort_devices_based_on_available_memory,
)
from transformer_lens.utilities.multi_gpu import get_device_for_block_index


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


def _cuda_cfg(n_layers: int, n_devices: int) -> SimpleNamespace:
    return SimpleNamespace(n_layers=n_layers, n_devices=n_devices, device="cuda")


class TestGetDeviceForBlockIndex:
    """Regression tests for the layer-to-device index math.

    Issue #1356: the previous formula ``index // (n_layers // n_devices)``
    overshot ``n_devices - 1`` whenever ``n_layers`` was not a multiple of
    ``n_devices``, and divided by zero when ``n_layers < n_devices``.
    """

    @pytest.mark.parametrize(
        "n_layers,n_devices",
        [(62, 8), (12, 8), (32, 4), (24, 8), (8, 8), (1, 8), (7, 8)],
    )
    def test_device_index_stays_in_bounds(self, n_layers: int, n_devices: int):
        cfg = _cuda_cfg(n_layers, n_devices)
        for index in range(n_layers):
            result = get_device_for_block_index(index, cfg)
            assert 0 <= result.index < n_devices, (
                f"index={index} mapped to device {result.index} which is outside "
                f"[0, {n_devices - 1}] for n_layers={n_layers}, n_devices={n_devices}"
            )

    @pytest.mark.parametrize("n_layers,n_devices", [(62, 8), (32, 4), (24, 8), (8, 8)])
    def test_layer_distribution_is_balanced(self, n_layers: int, n_devices: int):
        """Every device sees ``floor(n_layers / n_devices)`` or that plus 1 layers
        — never more than 1 layer off, and the counts sum to ``n_layers``."""
        cfg = _cuda_cfg(n_layers, n_devices)
        counts = [0] * n_devices
        for index in range(n_layers):
            counts[get_device_for_block_index(index, cfg).index] += 1
        assert sum(counts) == n_layers
        assert max(counts) - min(counts) <= 1, f"unbalanced distribution: {counts}"

    def test_first_index_lands_on_first_device(self):
        cfg = _cuda_cfg(n_layers=62, n_devices=8)
        result = get_device_for_block_index(0, cfg)
        assert result.index == 0

    def test_last_index_lands_on_last_device(self):
        cfg = _cuda_cfg(n_layers=62, n_devices=8)
        result = get_device_for_block_index(61, cfg)
        assert result.index == 7

    def test_starting_device_offset_is_honored(self):
        """When ``device`` carries an explicit index, layer offsets are added on top."""
        cfg = _cuda_cfg(n_layers=32, n_devices=4)
        result = get_device_for_block_index(0, cfg, device=torch.device("cuda", 2))
        assert result.index == 2
        result = get_device_for_block_index(31, cfg, device=torch.device("cuda", 2))
        assert result.index == 5  # 2 (starting offset) + 3 (last layer on 4 devices)

    def test_cpu_device_is_returned_unchanged(self):
        cfg = _cuda_cfg(n_layers=62, n_devices=8)
        result = get_device_for_block_index(30, cfg, device="cpu")
        assert result.type == "cpu"
