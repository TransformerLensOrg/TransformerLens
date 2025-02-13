"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

import transformer_lens

AvailableDeviceMemory = list[tuple[int, int]]
"""
This type is passed around between different CUDA memory operations.
The first entry of each tuple will be the device index.
The second entry will be how much memory is currently available.
"""


def calculate_available_device_cuda_memory(i: int) -> int:
    """Calculates how much memory is available at this moment for the device at the indicated index

    Args:
        i (int): The index we are looking at

    Returns:
        int: How memory is available
    """
    total = torch.cuda.get_device_properties(i).total_memory
    allocated = torch.cuda.memory_allocated(i)
    return total - allocated


def determine_available_memory_for_available_devices(max_devices: int) -> AvailableDeviceMemory:
    """Gets all available CUDA devices with their current memory calculated

    Returns:
        AvailableDeviceMemory: The list of all available devices with memory precalculated
    """
    devices = []
    for i in range(max_devices):
        devices.append((i, calculate_available_device_cuda_memory(i)))

    return devices


def sort_devices_based_on_available_memory(devices: AvailableDeviceMemory) -> AvailableDeviceMemory:
    """Sorts all available devices with devices with the most available memory returned first

    Args:
        devices (AvailableDeviceMemory): All available devices with memory calculated

    Returns:
        AvailableDeviceMemory: The same list of passed through devices sorted with devices with most
        available memory first
    """
    return sorted(devices, key=lambda x: x[1], reverse=True)


def get_best_available_cuda_device(max_devices: Optional[int] = None) -> torch.device:
    """Gets whichever cuda device has the most available amount of memory for use

    Raises:
        EnvironmentError: If there are no available devices, this will error out

    Returns:
        torch.device: The specific device that should be used
    """
    max_devices = max_devices if max_devices is not None else torch.cuda.device_count()
    devices = determine_available_memory_for_available_devices(max_devices)

    if len(devices) <= 0:
        raise EnvironmentError(
            "TransformerLens has been configured to use CUDA, but no available devices are present"
        )

    sorted_devices = sort_devices_based_on_available_memory(devices=devices)

    return torch.device("cuda", sorted_devices[0][0])


def get_best_available_device(cfg: "transformer_lens.HookedTransformerConfig") -> torch.device:
    """Gets the best available device to be used based on the passed in arguments

    Args:
        device (Union[torch.device, str]): Either the existing torch device or the string identifier

    Returns:
        torch.device: The best available device
    """
    assert cfg.device is not None
    device = torch.device(cfg.device)

    if device.type == "cuda":
        return get_best_available_cuda_device(cfg.n_devices)
    else:
        return device


def get_device_for_block_index(
    index: int,
    cfg: "transformer_lens.HookedTransformerConfig",
    device: Optional[Union[torch.device, str]] = None,
):
    """
    Determine the device for a given layer index based on the model configuration.

    This function assists in distributing model layers across multiple devices. The distribution
    is based on the configuration's number of layers (cfg.n_layers) and devices (cfg.n_devices).


    Args:
        index (int): Model layer index.
        cfg (HookedTransformerConfig): Model and device configuration.
        device (Optional[Union[torch.device, str]], optional): Initial device used for determining the target device.
            If not provided, the function uses the device specified in the configuration (cfg.device).

    Returns:
        torch.device: The device for the specified layer index.

    Deprecated:
        This function did not take into account a few factors for multi-GPU support. You should now
        use get_best_available_device in order to properly run models on multiple devices.
        This will be removed in 3.0
    """
    assert cfg.device is not None
    layers_per_device = cfg.n_layers // cfg.n_devices
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    device_index = (device.index or 0) + (index // layers_per_device)
    return torch.device(device.type, device_index)


def move_to_and_update_config(
    model: Union[
        "transformer_lens.HookedTransformer",
        "transformer_lens.HookedEncoder",
        "transformer_lens.HookedEncoderDecoder",
    ],
    device_or_dtype: Union[torch.device, str, torch.dtype],
    print_details=True,
):
    """
    Wrapper around `to` that also updates `model.cfg`.
    """
    if isinstance(device_or_dtype, torch.device):
        model.cfg.device = device_or_dtype.type
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, str):
        model.cfg.device = device_or_dtype
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, torch.dtype):
        model.cfg.dtype = device_or_dtype
        if print_details:
            print("Changing model dtype to", device_or_dtype)
        # change state_dict dtypes
        for k, v in model.state_dict().items():
            model.state_dict()[k] = v.to(device_or_dtype)
    return nn.Module.to(model, device_or_dtype)
