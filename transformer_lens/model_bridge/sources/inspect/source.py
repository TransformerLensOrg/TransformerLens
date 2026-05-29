"""``boot_inspect`` — wrap an ``inspect_ai`` provider in a RemoteBridge via InspectDriver."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import torch

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
    detect_tokenizer_bos_eos,
)
from transformer_lens.model_bridge.sources._hf_format import setup_tokenizer
from transformer_lens.utilities.hf_utils import get_hf_token

from . import hooks, profiles
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
    ``tl_bridge`` provider (residual/attn/mlp capture + full affine interventions +
    full-sequence logits, parity with ``boot_transformers`` on the verified
    architectures); pass ``"vllm-lens"`` to consume a running vllm-lens vLLM provider
    instead (residual-only, additive-steering-only). (``transformer_lens`` is
    inspect_ai's own built-in provider — a different thing — so ours is ``tl_bridge``.)

    Fireable hooks (``tl_bridge``, TransformerBridge-native names matching the vLLM
    driver): ``blocks.{i}.hook_in`` (resid_pre) / ``ln2.hook_in`` (resid_mid) /
    ``hook_out`` (resid_post) / ``attn.hook_out`` / ``mlp.hook_out``. Head-split hooks
    (q/k/v/z, pattern), ``embed``, and ``ln_final`` are non-fireable — use
    ``boot_transformers()`` for those.

    Full-sequence logits ride on ``return_logits=True`` (the default) — needed for
    logits/loss parity. For pure activation capture, pass ``return_logits=False`` to
    skip the (seq × d_vocab) logits payload; ``run_with_cache`` keeps them since it
    returns logits. Parity is verified only for ``hooks.PARITY_VERIFIED_ARCHITECTURES``
    (boot warns otherwise).
    """
    from inspect_ai.model import get_model
    from transformers import AutoConfig, AutoTokenizer

    from . import provider as _provider  # noqa: F401 — import registers our @modelapi

    hf_token = get_hf_token()
    hf_config = AutoConfig.from_pretrained(model_name, token=hf_token)
    architecture = hf_config.architectures[0]
    resolved_dtype = dtype or _dtype_from_hf_config(hf_config)

    if provider == "tl_bridge" and architecture not in hooks.PARITY_VERIFIED_ARCHITECTURES:
        warnings.warn(
            f"Hook parity vs boot_transformers is unverified for {architecture!r} "
            f"(verified: {sorted(hooks.PARITY_VERIFIED_ARCHITECTURES)}). The residual/attn/mlp "
            "boundaries assume the standard HF decoder-layer layout; diff a few hooks against "
            "boot_transformers before trusting captured values.",
            UserWarning,
            stacklevel=2,
        )

    bridge_config = build_bridge_config_from_hf(hf_config, architecture, model_name, resolved_dtype)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    # Match boot_transformers' tokenizer setup so to_tokens(str) is token-identical.
    tokenizer = setup_tokenizer(
        tokenizer, default_padding_side=getattr(adapter.cfg, "default_padding_side", None)
    )
    (
        adapter.cfg.tokenizer_prepends_bos,
        adapter.cfg.tokenizer_appends_eos,
    ) = detect_tokenizer_bos_eos(tokenizer)

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
