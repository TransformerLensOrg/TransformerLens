from unittest.mock import Mock

import torch

from transformer_lens.utilities.devices import (
    calculate_available_device_cuda_memory,
    determine_available_memory_for_available_devices,
    sort_devices_based_on_available_memory,
)


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
