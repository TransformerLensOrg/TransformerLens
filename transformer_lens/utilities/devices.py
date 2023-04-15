from typing import Optional, Union

import torch
from transformer_lens import HookedTransformerConfig


def get_device_for_block_index(
    index: int,
    cfg: HookedTransformerConfig,
    device: Optional[Union[torch.device, str]] = None,
):
    assert cfg.device is not None
    layers_per_device = cfg.n_layers // cfg.n_devices
    if device is None:
        device = cfg.device
    if isinstance(device, torch.device):
        device = device.type
    return torch.device(device, index // layers_per_device)
