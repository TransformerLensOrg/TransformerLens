"""``boot_vllm`` — construct a vLLM LLM, wrap it in a RemoteBridge via VLLMDriver."""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, Optional

import torch

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources._hf_format import (
    map_default_transformer_lens_config,
)
from transformer_lens.utilities.hf_utils import get_hf_token

from . import plugin
from .driver import VLLMDriver
from .internals import extract_hf_config, extract_inner_model, walk_dot_path
from .overlays import get_overlay

# Forced LLM(...) kwargs locked by the v1 plan. Caller override → ValueError.
_LOCKED_KWARGS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "skip_tokenizer_init": True,
    "disable_log_stats": True,
}


def boot_vllm(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: int = 2048,
    **vllm_kwargs: Any,
) -> RemoteBridge:
    """Boot a model via vLLM and wrap it in a :class:`RemoteBridge` via :class:`VLLMDriver`.

    vLLM drives the forward pass (PagedAttention + ``torch.compile`` + CUDA graphs).
    Capture buffers are populated by hooks the plugin installs pre-compile inside
    the worker; they come back via ``collective_rpc`` and replay through the
    bridge's HookPoint tree.
    """
    _reject_locked_overrides(vllm_kwargs)

    from transformers import AutoConfig, AutoTokenizer

    # Resolve architecture WITHOUT loading weights so we can tell the plugin
    # which dot-paths to hook before LLM(...) constructs the worker.
    hf_token = get_hf_token()
    hf_config_preview = AutoConfig.from_pretrained(model_name, token=hf_token)
    architecture = hf_config_preview.architectures[0]
    overlay = get_overlay(architecture)

    resolved_dtype = dtype or _dtype_from_hf_config(hf_config_preview)
    plugin.configure(
        capture_specs=overlay.capture_specs(hf_config_preview),
        max_num_batched_tokens=max_num_batched_tokens,
        dtype=resolved_dtype,
    )
    plugin.register()

    # The plugin's _config singleton must be visible to the worker, which only
    # holds with single-process execution. Multi-GPU is a v2 plan. Override
    # any user setting — silent capture failure otherwise.
    existing_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if existing_mp not in (None, "0"):
        warnings.warn(
            f"VLLM_ENABLE_V1_MULTIPROCESSING={existing_mp!r} overridden to '0' — "
            "boot_vllm needs single-process execution for capture hooks to install. "
            "Multi-GPU vLLM is a v2 plan.",
            UserWarning,
            stacklevel=2,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=str(resolved_dtype).replace("torch.", "") if dtype is not None else "auto",
        **_LOCKED_KWARGS,
        **vllm_kwargs,
    )

    hf_config = extract_hf_config(llm)
    _validate_capture_paths(llm, overlay, hf_config, architecture)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Build the adapter (RemoteBridge skips adapter.prepare_model — there's no
    # local model tree to walk).
    bridge_config = map_default_transformer_lens_config(hf_config)
    bridge_config.architecture = architecture
    bridge_config.model_name = model_name
    bridge_config.dtype = resolved_dtype
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)

    driver = VLLMDriver(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        overlay=overlay,
        hf_config=hf_config,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    bridge = RemoteBridge(adapter=adapter, tokenizer=tokenizer, driver=driver)
    _warn_nonfiring_hooks(model_name, architecture, overlay, hf_config)
    return bridge


def _validate_capture_paths(llm: Any, overlay: Any, hf_config: Any, architecture: str) -> None:
    """Walk each capture-spec dot-path against the loaded model. Raise early with
    a clear message if any are missing — otherwise the failure surfaces deep in
    the worker's patched_load_model with an opaque AttributeError.
    """
    inner = extract_inner_model(llm)
    specs = overlay.capture_specs(hf_config)
    missing = []
    for canonical_name, (dot_path, _width) in specs.items():
        try:
            walk_dot_path(inner, dot_path)
        except (AttributeError, IndexError, KeyError, ValueError):
            missing.append((canonical_name, dot_path))
    if missing:
        details = "\n".join(f"  {name} → {path}" for name, path in missing)
        raise NotImplementedError(
            f"{architecture} doesn't match the vLLM decoder-only layout the overlay "
            f"expects ({len(missing)} dot-path(s) missing):\n{details}\n"
            "Use boot_transformers() for this architecture, or add a custom overlay."
        )


def _reject_locked_overrides(vllm_kwargs: Dict[str, Any]) -> None:
    for key, locked in _LOCKED_KWARGS.items():
        if key in vllm_kwargs and vllm_kwargs[key] != locked:
            raise ValueError(
                f"boot_vllm forces {key}={locked}; caller passed {key}={vllm_kwargs[key]}. "
                "Multi-device / continuous batching / vLLM-owned tokenizer are v2 scope."
            )


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _warn_nonfiring_hooks(model_name: str, architecture: str, overlay: Any, hf_config: Any) -> None:
    non_firing = overlay.nonfiring_hooks()
    if not non_firing:
        return
    n_layers = getattr(hf_config, "num_hidden_layers", None)
    if isinstance(n_layers, int) and n_layers > 0:
        expanded = [name.replace("{i}", f"0..{n_layers - 1}") for name in non_firing]
    else:
        expanded = list(non_firing)
    logging.getLogger("transformer_lens.vllm").info(
        "vLLM source on %s (%s): the following hooks will not fire (vLLM fuses these): %s. "
        "Use boot_transformers() if you need them.",
        model_name,
        architecture,
        ", ".join(expanded),
    )


__all__ = ["boot_vllm"]
