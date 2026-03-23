"""Basic device utilities.

Simple utilities for moving models to devices and updating their configurations.
"""

from __future__ import annotations

from typing import Any, Protocol, Union, runtime_checkable

import torch
from torch import nn


def get_device() -> torch.device:
    """Get the best available device for the current system.

    Returns:
        torch.device: The best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return torch.device("mps")

    return torch.device("cpu")


@runtime_checkable
class ModelWithCfg(Protocol):
    """Protocol for models that have a config attribute and can be moved to devices."""

    cfg: Any

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the model's state dictionary."""
        ...

    def to(self, device_or_dtype: Union[torch.device, str, torch.dtype]) -> Any:
        """Move the model to a device or change its dtype."""
        ...


def move_to_and_update_config(
    model: ModelWithCfg,
    device_or_dtype: Union[torch.device, str, torch.dtype],
    print_details: bool = True,
) -> Any:
    """
    Wrapper around `to` that also updates `model.cfg`.

    Args:
        model: The model to move/update
        device_or_dtype: Device or dtype to move/change to
        print_details: Whether to print details about the operation

    Returns:
        The model after the operation
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
        # Update dtype in config if it exists
        if hasattr(model.cfg, "dtype"):
            model.cfg.dtype = device_or_dtype
        if print_details:
            print("Changing model dtype to", device_or_dtype)
        # change state_dict dtypes
        for k, v in model.state_dict().items():
            model.state_dict()[k] = v.to(device_or_dtype)

    # Call the base nn.Module.to() method to avoid recursion with custom to() methods
    # Use the unbound method approach to avoid calling the overridden to() method
    return nn.Module.to(model, device_or_dtype)  # type: ignore
