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
from .internals import extract_hf_config
from .overlays import get_overlay

# Forced LLM(...) kwargs that the capture-hook design depends on. Caller override → ValueError.
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

    **Scope vs vllm-lens:** vllm-lens is observation-only. This source extends to
    observation + spec-vocabulary *mutation* — each capture hook also applies an
    affine transform ``output = output * scale + bias`` (default identity), so
    interventions (``suppress`` / ``scale`` / ``add`` / ``set``) propagate to
    downstream layers. The hook's return value replaces the module output per
    PyTorch ``register_forward_hook`` semantics. **The mutation path under
    torch.compile + CUDA graphs is verified end-to-end by**
    ``demos/vLLM_Bridge_Integration_Test.ipynb``; unit tests cover the dispatch
    protocol only.

    GPU memory cost: each capture buffer is ``max_num_batched_tokens × width`` at
    the model's dtype. For Llama-3.2-1B at fp16 with ``max_num_batched_tokens=2048``,
    the unembed buffer alone is ~525 MB (2048 × 128256 × 2 bytes); residual-stream
    buffers add ~8 MB per hook. The affine intervention hook also allocates a
    transient output-shape tensor per forward (even in identity mode), so peak
    forward memory is ~1.5× the capture buffers' resident size. Plan accordingly.
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
    # holds with single-process execution. Multi-GPU is unsupported. Override
    # any user setting — silent capture failure otherwise.
    existing_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if existing_mp not in (None, "0"):
        warnings.warn(
            f"VLLM_ENABLE_V1_MULTIPROCESSING={existing_mp!r} overridden to '0' — "
            "boot_vllm needs single-process execution for capture hooks to install. "
            "Multi-GPU vLLM is unsupported.",
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

    # Capture-path validity is enforced inside patched_load_model during
    # LLM(...) above — any missing dot-path raises AttributeError there. By
    # the time we reach this line every spec has already been walked successfully.
    hf_config = extract_hf_config(llm)
    # _config has been consumed into per-Worker state; clear so a non-TL
    # vllm.LLM(...) in the same process doesn't inherit our specs.
    plugin._config.clear()
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
    _log_hook_summary(model_name, architecture, driver)
    return bridge


def _reject_locked_overrides(vllm_kwargs: Dict[str, Any]) -> None:
    for key, locked in _LOCKED_KWARGS.items():
        if key in vllm_kwargs and vllm_kwargs[key] != locked:
            raise ValueError(
                f"boot_vllm forces {key}={locked}; caller passed {key}={vllm_kwargs[key]}. "
                "Multi-device / continuous batching / vLLM-owned tokenizer are unsupported."
            )


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _log_hook_summary(model_name: str, architecture: str, driver: VLLMDriver) -> None:
    """Log the fireable and non-fireable hook sets so users don't have to grep."""
    log = logging.getLogger("transformer_lens.vllm")
    fireable = sorted(driver.supported_hook_points)
    log.info(
        "vLLM source on %s (%s) captures %d hook(s): %s",
        model_name, architecture, len(fireable), ", ".join(fireable),
    )
    nonfiring = sorted(driver.non_fireable_hook_points)
    if nonfiring:
        log.info(
            "vLLM source on %s (%s) cannot fire %d hook(s) (vLLM fuses these): %s. "
            "Use boot_transformers() if you need them.",
            model_name, architecture, len(nonfiring), ", ".join(nonfiring),
        )


__all__ = ["boot_vllm"]
