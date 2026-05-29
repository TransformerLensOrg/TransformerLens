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
    full-sequence logits); ``"vllm-lens"`` targets a running vllm-lens vLLM provider
    (residual-only, additive-steering-only) — wire-aligned with its documented format,
    but not yet verified against a live provider.

    Fireable hooks (``tl_bridge``, TransformerBridge-native names): ``blocks.{i}.hook_in``
    (resid_pre) / ``ln2.hook_in`` (resid_mid) /
    ``hook_out`` (resid_post) / ``attn.hook_out`` / ``mlp.hook_out``. The provider runs
    a structural self-check per model and gates any boundary it can't serve faithfully:
    ``resid_mid`` for parallel-residual or norm-variant blocks, and ``attn_out``/
    ``mlp_out`` when their submodule isn't locatable (it warns when it gates one).
    Head-split hooks (q/k/v/z, pattern), ``embed``, and ``ln_final`` are always
    non-fireable — use ``boot_transformers()`` for those.

    For parity with ``boot_transformers`` the provider loads with the same dtype (fp32 by
    default) and eager attention. Full-sequence logits ride on ``return_logits=True`` (the
    default); pass ``return_logits=False`` to skip the (seq × d_vocab) payload for pure
    activation capture (``run_with_cache`` keeps them since it returns logits).
    """
    from inspect_ai.model import get_model
    from transformers import AutoConfig, AutoTokenizer

    from . import provider as _provider  # noqa: F401 — import registers our @modelapi

    hf_token = get_hf_token()
    hf_config = AutoConfig.from_pretrained(model_name, token=hf_token)
    architecture = hf_config.architectures[0]
    # Default fp32 to match boot_transformers (which loads/casts fp32 regardless of the
    # config's native dtype); an explicit dtype still wins.
    resolved_dtype = dtype if dtype is not None else torch.float32

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

    if provider == "tl_bridge":
        # The provider's raw HF forward must match boot_transformers' load: same dtype,
        # eager attention (TL forces eager — SDPA/flash diverge and accumulate with depth),
        # and auth/remote-code so gated/custom models load at all.
        inspect_kwargs["model_kwargs"] = _provider_model_kwargs(
            dict(inspect_kwargs.get("model_kwargs", {})), adapter, resolved_dtype, hf_token
        )

    # memoize=False: inspect_ai caches get_model by name, which would (a) return a stale
    # model ignoring a changed dtype/kwargs on re-boot and (b) keep weights resident past
    # close(). Each boot must honor its own args and own its model's lifecycle.
    model = get_model(f"{provider}/{model_name}", memoize=False, **inspect_kwargs)
    # For our HF provider, restrict the profile to the boundaries its structural self-check
    # found this model can serve; warn only if it gated something.
    if provider == "tl_bridge":
        api = getattr(model, "api", None)
        kinds = None
        note = ""
        if api is not None:
            kinds = api.supported_kinds() if hasattr(api, "supported_kinds") else None
            note = api.capability_note() if hasattr(api, "capability_note") else ""
        profile = profiles.TLBridgeProfile(supported_kinds=kinds)
        if note:
            warnings.warn(note, UserWarning, stacklevel=2)
    else:
        profile = profiles.for_provider(provider)
    driver = InspectDriver(model=model, adapter=adapter, tokenizer=tokenizer, profile=profile)
    bridge = RemoteBridge(adapter=adapter, tokenizer=tokenizer, driver=driver)
    _log_hook_summary(model_name, architecture, provider, driver)
    return bridge


def _provider_model_kwargs(
    model_kwargs: dict[str, Any], adapter: Any, dtype: torch.dtype, hf_token: Optional[str]
) -> dict[str, Any]:
    """Load kwargs for the HF provider that mirror boot_transformers, so the provider's
    raw forward matches the bridge. Caller-supplied keys win (setdefault)."""
    model_kwargs.setdefault("torch_dtype", dtype)
    # boot_transformers forces eager unless the adapter pins an implementation.
    model_kwargs.setdefault(
        "attn_implementation", getattr(adapter.cfg, "attn_implementation", None) or "eager"
    )
    if hf_token:
        model_kwargs.setdefault("token", hf_token)
    return model_kwargs


def _log_hook_summary(
    model_name: str, architecture: str, provider: str, driver: InspectDriver
) -> None:
    log = logging.getLogger("transformer_lens.inspect")
    fireable = sorted(driver.supported_hook_points)
    log.info(
        "Inspect source on %s (%s) via provider %r serves %d fireable hook(s).",
        model_name,
        architecture,
        provider,
        len(fireable),
    )


__all__ = ["boot_inspect"]
