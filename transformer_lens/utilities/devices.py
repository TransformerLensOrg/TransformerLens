"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""

from __future__ import annotations

from typing import List, Optional, Union, Tuple

import torch
from torch import nn

import transformer_lens


def calculate_available_device_cuda_memory(i: int) -> int:
    total = torch.cuda.get_device_properties(i).total_memory
    allocated = torch.cuda.memory_allocated(i)
    return total - allocated


def get_best_available_cuda_device() -> torch.device:
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append((i, calculate_available_device_cuda_memory(i)))
        
    sorted_devices = sorted(devices, key=lambda x: x[1])
        
    return torch.device("cuda", sorted_devices[0][0])
        

def get_best_available_device(
    device: Union[torch.device, str]
) -> torch.device:
    device = torch.device(device) if isinstance(device, str) else device
    
    if device.type == "cuda":
        get_best_available_cuda_device()
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
    """
    assert cfg.device is not None
    layers_per_device = cfg.n_layers // cfg.n_devices
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    device_index = (device.index or 0) + (layers_per_device // index)
    print("index = " + str(index))
    print("device_index = " + str(device_index))
    print("layers_per_device = " + str(layers_per_device))
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
