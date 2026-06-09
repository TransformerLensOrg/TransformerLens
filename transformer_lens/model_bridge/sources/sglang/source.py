"""``boot_sglang`` — wrap an SGLang Engine in a RemoteBridge via SGLangDriver.

Build forward_hooks JSON specs from the overlay, bind PULL, construct Engine
(SGLang's spawned scheduler subprocess runs ``load_plugins()`` then
``register_forward_hooks`` against our factory). Hooks install AFTER full CUDA
graph capture so leaving graphs on silently bypasses captures during replay —
``disable_cuda_graph`` defaults to True."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.utilities.hf_utils import get_hf_token

from . import wire
from .capture_puller import CapturePuller
from .driver import SGLangDriver
from .internals import assert_sglang_supported, extract_hf_config
from .overlays import get_overlay

# Caller override → ValueError. Multi-device splits the model across worker
# subprocesses our single Scheduler patch can't address; we own the tokenizer.
# chunked_prefill_size=-1: chunked prefill would emit N partial messages per hook
# on the PULL socket, which the collector would silently truncate to one.
_LOCKED_KWARGS = {
    "tp_size": 1,
    "dp_size": 1,
    "skip_tokenizer_init": True,
    "chunked_prefill_size": -1,
}


def boot_sglang(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    mem_fraction_static: float = 0.5,
    max_total_tokens: Optional[int] = None,
    max_num_batched_tokens: int = 2048,
    disable_cuda_graph: bool = True,
    **sglang_kwargs: Any,
) -> RemoteBridge:
    """Boot a model via SGLang; returns a :class:`RemoteBridge` wrapping :class:`SGLangDriver`.

    Returned logits are sampler log-probs at position -1 only (argmax-correct,
    absolute scale off); ``return_type='loss'|'both'`` is rejected. Set
    ``disable_cuda_graph=False`` only if you don't need captures inside graphed forwards.
    """
    _reject_locked_overrides(sglang_kwargs)
    assert_sglang_supported()

    from transformers import AutoConfig, AutoTokenizer

    hf_token = get_hf_token()
    hf_config_preview = AutoConfig.from_pretrained(model_name, token=hf_token)
    architecture = hf_config_preview.architectures[0]
    overlay = get_overlay(architecture)
    resolved_dtype = dtype or _dtype_from_hf_config(hf_config_preview)

    capture_specs = overlay.capture_specs(hf_config_preview)
    channel = wire.fresh_channel()
    forward_hooks = wire.build_forward_hooks(capture_specs, channel)

    # Bind PULL before Engine spawns — kills the connect-before-bind race.
    puller = CapturePuller(channel)

    from sglang.srt.entrypoints.engine import Engine  # type: ignore[import-not-found]

    # max_num_batched_tokens stays driver-internal as the prompt-length gate;
    # ServerArgs uses chunked_prefill_size / max_prefill_tokens instead.
    engine_kwargs: Dict[str, Any] = {
        "model_path": model_name,
        "mem_fraction_static": mem_fraction_static,
        "max_total_tokens": max_total_tokens,
        "dtype": _dtype_arg_for_sglang(resolved_dtype, explicit=dtype is not None),
        "forward_hooks": forward_hooks,
        "disable_cuda_graph": disable_cuda_graph,
    }
    engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}

    try:
        engine = Engine(**engine_kwargs, **_LOCKED_KWARGS, **sglang_kwargs)
    except Exception:
        # Engine ctor may fail (auth, CUDA OOM, version drift). Free the PULL
        # binding before propagating so the caller can retry on the same channel.
        puller.close()
        raise

    hf_config = extract_hf_config(engine)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    bridge_config = build_bridge_config_from_hf(hf_config, architecture, model_name, resolved_dtype)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)

    driver = SGLangDriver(
        engine=engine,
        adapter=adapter,
        tokenizer=tokenizer,
        overlay=overlay,
        hf_config=hf_config,
        max_num_batched_tokens=max_num_batched_tokens,
        puller=puller,
    )
    bridge = RemoteBridge(adapter=adapter, tokenizer=tokenizer, driver=driver)
    _log_hook_summary(model_name, architecture, driver)
    return bridge


def _reject_locked_overrides(sglang_kwargs: Dict[str, Any]) -> None:
    for key, locked in _LOCKED_KWARGS.items():
        if key in sglang_kwargs and sglang_kwargs[key] != locked:
            raise ValueError(
                f"boot_sglang forces {key}={locked}; caller passed {key}={sglang_kwargs[key]}."
            )


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _dtype_arg_for_sglang(dtype: torch.dtype, *, explicit: bool) -> Optional[str]:
    """``"float16"`` style for SGLang; ``"auto"`` when caller didn't pin."""
    if not explicit:
        return "auto"
    return str(dtype).replace("torch.", "")


def _log_hook_summary(model_name: str, architecture: str, driver: SGLangDriver) -> None:
    log = logging.getLogger("transformer_lens.sglang")
    fireable = sorted(driver.supported_hook_points)
    log.info(
        "SGLang source on %s (%s) captures %d hook(s).",
        model_name,
        architecture,
        len(fireable),
    )


__all__ = ["boot_sglang"]
