from typing import Optional, Union

import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def get_device_for_block_index(
    index: int,
    cfg: HookedTransformerConfig,
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
    device_index = (device.index or 0) + (index // layers_per_device)
    return torch.device(device.type, device_index)
