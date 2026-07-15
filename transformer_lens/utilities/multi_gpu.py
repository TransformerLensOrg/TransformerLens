"""Multi-GPU utilities.

Utilities for managing multiple GPU devices and distributing model layers across them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

if TYPE_CHECKING:
    from transformer_lens.config.hooked_transformer_config import (
        HookedTransformerConfig as ConfigType,
    )
else:
    ConfigType = Any

_UNSUPPORTED_OFFLOAD_DEVICE_MAP_VALUES = {"disk"}

AvailableDeviceMemory = list[tuple[int, int]]
"""
This type is passed around between different CUDA memory operations.
The first entry of each tuple will be the device index.
The second entry will be how much memory is currently available.
"""

MaxMemory = Dict[Union[str, int], Union[str, int]]


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


def _get_available_cuda_memory_for_device(i: int) -> int:
    """Return a concrete byte budget accepted by Accelerate's ``max_memory``."""
    try:
        free_memory, _ = torch.cuda.mem_get_info(i)
    except (AttributeError, RuntimeError):
        return calculate_available_device_cuda_memory(i)
    return int(free_memory)


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


def get_best_available_device(
    cfg: ConfigType,
) -> torch.device:
    """Gets the best available device to be used based on the passed in arguments

    Args:
        cfg: The HookedTransformerConfig object containing device configuration

    Returns:
        torch.device: The best available device
    """
    assert cfg.device is not None
    device = torch.device(cfg.device)

    if device.type == "cuda" and cfg.n_devices > 1:
        return get_best_available_cuda_device(cfg.n_devices)
    else:
        return device


def get_device_for_block_index(
    index: int,
    cfg: ConfigType,
    device: Optional[Union[torch.device, str]] = None,
):
    """
    Determine the device for a given layer index based on the model configuration.

    This function assists in distributing model layers across multiple devices. The distribution
    is based on the configuration's number of layers (cfg.n_layers) and devices (cfg.n_devices).


    Args:
        index (int): Model layer index.
        cfg: Model and device configuration.
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
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == "cpu":
        return device
    # Multiplying first guarantees the result is in [0, n_devices - 1] and avoids
    # the divide-by-zero when n_layers < n_devices. The naive form
    # `index // (n_layers // n_devices)` floors the divisor and overshoots when
    # n_layers is not a multiple of n_devices (e.g. 62 layers / 8 devices → 8).
    device_index = (device.index or 0) + (index * cfg.n_devices) // cfg.n_layers
    return torch.device(device.type, device_index)


def resolve_device_map(
    n_devices: Optional[int],
    device_map: Optional[Union[str, Dict[str, Union[str, int]]]],
    device: Optional[Union[str, torch.device]],
    max_memory: Optional[MaxMemory] = None,
) -> Tuple[Optional[Union[str, Dict[str, Union[str, int]]]], Optional[MaxMemory]]:
    """Resolve ``n_devices`` / ``device_map`` / ``device`` into HF ``from_pretrained`` kwargs.

    Returns ``(device_map, max_memory)`` tuple ready to pass into ``model_kwargs``.

    Semantics:
        - Explicit ``device_map`` wins and is passed through unchanged (user-provided
          ``max_memory`` is passed through too). CPU targets are supported; disk / meta
          offload targets are still rejected because Bridge component wrappers can bypass
          Accelerate's offload hooks during forward passes.
        - ``n_devices=None`` or ``1``: returns ``(None, None)`` — single-device path.
        - ``n_devices > 1``: returns ``("balanced", {0: bytes, ..., n-1: bytes})``.
          ``"balanced"`` is accelerate's string directive for balanced layer dispatch;
          concrete byte budgets cap visibility to exactly ``n_devices`` GPUs.
    """
    if device_map is not None and device is not None:
        raise ValueError("device and device_map are mutually exclusive — pass one.")
    if device_map is not None:
        _validate_device_map_values(device_map)
        return device_map, max_memory
    if n_devices is None or n_devices <= 1:
        return None, max_memory
    if not torch.cuda.is_available():
        raise ValueError(f"n_devices={n_devices} requires CUDA, which is not available.")
    if torch.cuda.device_count() < n_devices:
        raise ValueError(
            f"n_devices={n_devices} but only {torch.cuda.device_count()} CUDA devices present."
        )
    resolved_max_memory: MaxMemory = (
        dict(max_memory)
        if max_memory is not None
        else {i: _get_available_cuda_memory_for_device(i) for i in range(n_devices)}
    )
    return "balanced", resolved_max_memory


def _validate_device_map_values(
    device_map: Union[str, Dict[str, Union[str, int]]],
) -> None:
    """Reject explicit disk values in a user-supplied device_map dict.
    Meta values are passed through (validated at boot against load_weights)."""
    if isinstance(device_map, str):
        return
    for key, value in device_map.items():
        normalized = str(value).lower() if isinstance(value, str) else None
        if normalized in _UNSUPPORTED_OFFLOAD_DEVICE_MAP_VALUES:
            raise ValueError(
                f"device_map[{key!r}]={value!r} is not supported yet. TransformerBridge "
                "currently supports CPU device_map targets, but disk / meta offload can "
                "bypass Accelerate hooks inside wrapped Bridge components."
            )


def cast_floating_params_to_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast materialized floating parameters while preserving Accelerate offload hooks."""
    from accelerate.utils import align_module_device

    for module in model.modules():
        with align_module_device(module):
            for param in module.parameters(recurse=False):
                if not param.is_floating_point() or param.dtype == dtype:
                    continue
                if param.device.type == "meta":
                    continue
                param.data = param.data.to(dtype=dtype)


def find_embedding_device(hf_model: Any) -> Optional[torch.device]:
    """Return the device that input tokens should be placed on for a dispatched HF model.

    When a model is loaded with ``device_map``, accelerate populates ``hf_device_map``
    and inserts pre/post-forward hooks that route activations. Input tensors must land on
    the device of whichever module first *consumes* them — the input embedding. Returns
    ``None`` for single-device models (no ``hf_device_map`` set).

    Resolves via ``hf_model.get_input_embeddings()`` rather than dict insertion order to
    cover encoder-decoder / multimodal / audio architectures where the first entry in
    ``hf_device_map`` is not the text-token embedding (e.g. the vision tower on LLaVA).
    """
    hf_device_map = getattr(hf_model, "hf_device_map", None)
    if not hf_device_map:
        return None
    # Preferred: ask the model for its input embedding module and read its device.
    get_input_embeddings = getattr(hf_model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        try:
            embed_module = get_input_embeddings()
        except (AttributeError, NotImplementedError):
            embed_module = None
        if embed_module is not None:
            try:
                param = next(embed_module.parameters())
                return param.device
            except StopIteration:
                pass
    # Fallback: first entry in hf_device_map. Less reliable but better than nothing.
    first_device = next(iter(hf_device_map.values()))
    if isinstance(first_device, int):
        return torch.device("cuda", first_device)
    return torch.device(first_device)


def count_unique_devices(hf_model: Any) -> int:
    """Count the number of unique devices across a dispatched HF model's ``hf_device_map``.

    Returns 1 if the model has no ``hf_device_map`` (single-device load).
    """
    hf_device_map = getattr(hf_model, "hf_device_map", None)
    if not hf_device_map:
        return 1
    return len(set(hf_device_map.values()))
