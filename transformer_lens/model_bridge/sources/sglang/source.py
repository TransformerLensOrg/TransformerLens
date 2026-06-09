"""``boot_sglang`` — construct an SGLang Engine, wrap it in a RemoteBridge via SGLangDriver."""
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

from . import plugin
from .driver import SGLangDriver
from .internals import assert_sglang_supported, extract_hf_config
from .overlays import get_overlay

# Engine kwargs the capture-hook design depends on; caller override → ValueError.
# Multi-device parallelism breaks the single-ModelRunner monkey-patch; we own tokenization
# on the bridge side for to_tokens consistency.
_LOCKED_KWARGS = {
    "tp_size": 1,
    "dp_size": 1,
    "skip_tokenizer_init": True,
}


def boot_sglang(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    mem_fraction_static: float = 0.5,
    max_total_tokens: Optional[int] = None,
    max_num_batched_tokens: int = 2048,
    **sglang_kwargs: Any,
) -> RemoteBridge:
    """Boot a model via SGLang; returns a :class:`RemoteBridge` wrapping :class:`SGLangDriver`.

    Mirrors :func:`boot_vllm` in shape. The plugin installs capture hooks in
    ``ModelRunner.load_model`` (pre-compile); affine interventions ride the same
    hooks. End-to-end verification is in ``demos/SGLang_Bridge_Integration_Test.ipynb``.

    Returned logits are sampler log-probs at position -1 only — argmax-correct,
    absolute scale off; ``return_type='loss'|'both'`` is rejected.

    Pass an explicit ``max_total_tokens`` (e.g. 2048) on smaller GPUs; the model's
    native context (e.g. 131072 for Llama-3.2-1B) reserves several GiB otherwise.
    """
    _reject_locked_overrides(sglang_kwargs)
    assert_sglang_supported()

    from transformers import AutoConfig, AutoTokenizer

    # Resolve architecture without loading weights so we can prime the plugin
    # before Engine(...) constructs the worker.
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

    from sglang.srt.entrypoints.engine import Engine  # type: ignore[import-not-found]

    engine_kwargs: Dict[str, Any] = {
        "model_path": model_name,
        "mem_fraction_static": mem_fraction_static,
        "max_total_tokens": max_total_tokens,
        "max_num_batched_tokens": max_num_batched_tokens,
        "dtype": _dtype_arg_for_sglang(resolved_dtype, explicit=dtype is not None),
    }
    # Drop None values so SGLang's defaults apply instead of explicit None.
    engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
    engine = Engine(**engine_kwargs, **_LOCKED_KWARGS, **sglang_kwargs)

    # patched_load_model has already walked every spec by now; any missing dot-path
    # raised AttributeError before we get here.
    hf_config = extract_hf_config(engine)
    # Don't leak our specs to a subsequent non-TL Engine in the same process.
    plugin._config.clear()
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
    )
    bridge = RemoteBridge(adapter=adapter, tokenizer=tokenizer, driver=driver)
    _log_hook_summary(model_name, architecture, driver)
    return bridge


def _reject_locked_overrides(sglang_kwargs: Dict[str, Any]) -> None:
    for key, locked in _LOCKED_KWARGS.items():
        if key in sglang_kwargs and sglang_kwargs[key] != locked:
            raise ValueError(
                f"boot_sglang forces {key}={locked}; caller passed {key}={sglang_kwargs[key]}. "
                "Multi-device parallelism and SGLang-owned tokenization are unsupported."
            )


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float16)
    return torch.float16


def _dtype_arg_for_sglang(dtype: torch.dtype, *, explicit: bool) -> Optional[str]:
    """``"float16"``-style string for SGLang; ``"auto"`` when caller didn't pin."""
    if not explicit:
        return "auto"
    return str(dtype).replace("torch.", "")


def _log_hook_summary(model_name: str, architecture: str, driver: SGLangDriver) -> None:
    """Log the fireable and non-fireable hook sets at boot."""
    log = logging.getLogger("transformer_lens.sglang")
    fireable = sorted(driver.supported_hook_points)
    log.info(
        "SGLang source on %s (%s) captures %d hook(s): %s",
        model_name,
        architecture,
        len(fireable),
        ", ".join(fireable),
    )
    nonfiring = sorted(driver.non_fireable_hook_points)
    if nonfiring:
        log.info(
            "SGLang source on %s (%s) cannot fire %d hook(s) (SGLang fuses these): %s. "
            "Use boot_transformers() if you need them.",
            model_name,
            architecture,
            len(nonfiring),
            ", ".join(nonfiring),
        )


__all__ = ["boot_sglang"]
