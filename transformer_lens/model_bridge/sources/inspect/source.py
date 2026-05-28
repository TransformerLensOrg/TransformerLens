"""``boot_inspect`` — wrap an ``inspect_ai`` provider in a RemoteBridge via InspectDriver."""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.utilities.hf_utils import get_hf_token

from . import profiles
from .driver import InspectDriver


def boot_inspect(
    model_name: str,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    provider: str = "tl_bridge",
    **inspect_kwargs: Any,
) -> RemoteBridge:
    """Boot a model via an ``inspect_ai`` provider and wrap it in a :class:`RemoteBridge`.

    The driver is provider-agnostic: ``provider`` defaults to our own HF-backed
    ``tl_bridge`` provider (residual-stream capture + full affine interventions,
    exact parity with ``boot_transformers``); pass ``"vllm-lens"`` to consume a
    running vllm-lens vLLM provider instead — same wire format, no dependency on
    their package. (``transformer_lens`` is inspect_ai's own built-in provider — a
    different thing — so ours is ``tl_bridge``.)

    **Returned logits are last-position only** (next-token); ``return_type`` in
    ``{"loss", "both"}`` is rejected. Capture is **residual-stream only**
    (``blocks.{i}.hook_resid_post``); attn/mlp/pattern hooks are non-fireable —
    use ``boot_transformers()`` for those.
    """
    from inspect_ai.model import get_model
    from transformers import AutoConfig, AutoTokenizer

    from . import provider as _provider  # noqa: F401 — import registers our @modelapi

    hf_token = get_hf_token()
    hf_config = AutoConfig.from_pretrained(model_name, token=hf_token)
    architecture = hf_config.architectures[0]
    resolved_dtype = dtype or _dtype_from_hf_config(hf_config)

    bridge_config = build_bridge_config_from_hf(hf_config, architecture, model_name, resolved_dtype)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    model = get_model(f"{provider}/{model_name}", **inspect_kwargs)
    # The profile carries everything provider-specific: hook set, logits behavior,
    # request schema, intervention translation. Ours = full; vllm-lens = residual-only.
    profile = profiles.for_provider(provider)
    driver = InspectDriver(model=model, adapter=adapter, tokenizer=tokenizer, profile=profile)
    bridge = RemoteBridge(adapter=adapter, tokenizer=tokenizer, driver=driver)
    _log_hook_summary(model_name, architecture, provider, driver)
    return bridge


def _dtype_from_hf_config(hf_config: Any) -> torch.dtype:
    raw = getattr(hf_config, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw, torch.float32)
    return torch.float32


def _log_hook_summary(
    model_name: str, architecture: str, provider: str, driver: InspectDriver
) -> None:
    log = logging.getLogger("transformer_lens.inspect")
    fireable = sorted(driver.supported_hook_points)
    log.info(
        "Inspect source on %s (%s) via provider %r captures %d residual-stream hook(s).",
        model_name,
        architecture,
        provider,
        len(fireable),
    )


__all__ = ["boot_inspect"]
