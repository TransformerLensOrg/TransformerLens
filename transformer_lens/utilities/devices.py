"""Device utilities.

Utilities for device detection (with MPS safety), moving models to devices,
and updating their configurations.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Protocol, Union, runtime_checkable

import torch
from torch import nn

# ---------------------------------------------------------------------------
# MPS safety state
# ---------------------------------------------------------------------------

_mps_warned = False

# MPS silent correctness issues are known in PyTorch <= 2.7.
# Bump this when a PyTorch release ships verified MPS fixes.
_MPS_MIN_SAFE_TORCH_VERSION: tuple[int, ...] | None = None

# torch 2.8.0 on MPS has an upstream bug where torch.nn.functional.linear
# produces incorrect results for non-contiguous tensors. This silently
# corrupts generate() output and attention computations. Fixed in 2.9.0.
# See: https://github.com/pytorch/pytorch/issues/161640
# See: https://github.com/TransformerLensOrg/TransformerLens/issues/1062
_MPS_BROKEN_TORCH_VERSIONS: tuple[tuple[int, ...], ...] = ((2, 8),)

_mps_broken_torch_warned = False


def _torch_version_tuple() -> tuple[int, ...]:
    """Parse torch.__version__ into a comparable tuple, ignoring pre-release suffixes."""
    return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def _torch_mps_has_known_broken_bug() -> bool:
    """True if the installed torch version has a known-broken MPS path.

    Distinct from the generic MPS-may-be-unreliable warning: these are specific,
    upstream-fixed bugs where output is silently wrong regardless of opt-in.
    """
    return _torch_version_tuple() in _MPS_BROKEN_TORCH_VERSIONS


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Get the best available device, with MPS safety checks.

    MPS is only auto-selected when the environment variable
    ``TRANSFORMERLENS_ALLOW_MPS=1`` is set **and** the installed PyTorch
    version is 2.0 or higher.

    Returns:
        str: The best available device name (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            # Only auto-select MPS when explicitly opted-in via env var
            if os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "") == "1":
                return "mps"
            logging.info(
                "MPS device available but not auto-selected due to known correctness issues "
                "(PyTorch %s). Set TRANSFORMERLENS_ALLOW_MPS=1 to override. See: "
                "https://github.com/TransformerLensOrg/TransformerLens/issues/1178",
                torch.__version__,
            )

    return "cpu"


def warn_if_mps(device: Union[str, torch.device]) -> None:
    """Emit a one-time warning if device is MPS and TRANSFORMERLENS_ALLOW_MPS is not set.

    Automatically suppressed when the installed PyTorch version meets or exceeds
    _MPS_MIN_SAFE_TORCH_VERSION (currently unset — no version is considered safe yet).

    Also emits a separate, stronger warning for known-broken torch versions on MPS
    (see _MPS_BROKEN_TORCH_VERSIONS). This warning fires even when the user has
    opted in via TRANSFORMERLENS_ALLOW_MPS=1, because the affected operations
    produce silently wrong outputs regardless of opt-in.
    """
    global _mps_warned, _mps_broken_torch_warned
    if isinstance(device, torch.device):
        device = device.type
    if not (isinstance(device, str) and device == "mps"):
        return

    # Known-broken torch versions always warn (can't be opted-out of).
    if _torch_mps_has_known_broken_bug() and not _mps_broken_torch_warned:
        _mps_broken_torch_warned = True
        warnings.warn(
            f"PyTorch {torch.__version__} has a known MPS bug that produces "
            "silently incorrect results (torch.nn.functional.linear on "
            "non-contiguous tensors). This corrupts generate() output and "
            "attention computations. Upgrade to torch >= 2.9.0. "
            "See: https://github.com/TransformerLensOrg/TransformerLens/issues/1062 "
            "and https://github.com/pytorch/pytorch/issues/161640",
            UserWarning,
            stacklevel=2,
        )

    if _mps_warned:
        return
    if (
        _MPS_MIN_SAFE_TORCH_VERSION is not None
        and _torch_version_tuple() >= _MPS_MIN_SAFE_TORCH_VERSION
    ):
        return
    if os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "") != "1":
        _mps_warned = True
        warnings.warn(
            "MPS backend may produce silently incorrect results (PyTorch "
            f"{torch.__version__}). "
            "Set TRANSFORMERLENS_ALLOW_MPS=1 to suppress this warning. "
            "See: https://github.com/TransformerLensOrg/TransformerLens/issues/1178",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Model protocol & move helper
# ---------------------------------------------------------------------------


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
    from transformer_lens.utilities import warn_if_mps

    if isinstance(device_or_dtype, torch.device):
        warn_if_mps(device_or_dtype)
        model.cfg.device = device_or_dtype.type
        if print_details:
            print("Moving model to device: ", model.cfg.device)
    elif isinstance(device_or_dtype, str):
        warn_if_mps(device_or_dtype)
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
