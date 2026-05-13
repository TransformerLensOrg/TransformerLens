"""External-model source for TransformerLens.

Provides :func:`boot_external`, which installs the TransformerBridge's hooks
on a *pre-loaded* ``nn.Module`` without going through HuggingFace's
``from_pretrained`` pipeline. Use this when you have a custom loader (vLLM,
MLX, raw torch checkpoint, GGUF → torch, etc.) and just want the bridge's
instrumentation contract on top of your model.

The bridge never moves, casts, or otherwise mutates the user-supplied model.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.bridge import TransformerBridge

# Shared helpers — single source of truth lives in sources/transformers.py
# and is reused here so the HF→bridge translation never forks.
from transformer_lens.model_bridge.sources.transformers import (
    build_bridge_config_from_hf,
    detect_tokenizer_bos_eos,
    setup_tokenizer,
)


def boot_external(
    model: nn.Module,
    architecture: str,
    hf_config: Optional[Any] = None,
    tl_config: Optional[TransformerBridgeConfig] = None,
    tokenizer: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Any] = None,
    model_name: str = "external",
) -> TransformerBridge:
    """Wrap a pre-loaded model in a TransformerBridge without going through HF.

    Use when your model is already loaded by a non-HF pipeline (vLLM, MLX,
    custom torch.hub, GGUF → torch, raw checkpoints, etc.) and you just need
    TransformerLens's hook surface on top.

    The bridge **never moves, casts, or mutates** the supplied model.

    Args:
        model: Any ``nn.Module`` whose submodule tree matches the adapter's
            expected dot-paths for ``architecture``. Use
            :meth:`TransformerBridge.diagnose_paths` as a pre-flight check.
        architecture: Architecture identifier (e.g. ``"GPT2LMHeadModel"``,
            ``"LlamaForCausalLM"``). Picks the adapter from
            :data:`SUPPORTED_ARCHITECTURES`.
        hf_config: Optional HF-style config. Translated via the same path
            :func:`boot` uses. Mutually exclusive with ``tl_config``.
        tl_config: Optional pre-built :class:`TransformerBridgeConfig`.
            Bypasses HF translation entirely. Mutually exclusive with
            ``hf_config``.
        tokenizer: Optional tokenizer. If supplied it goes through
            :func:`setup_tokenizer`, which requires
            ``PreTrainedTokenizerBase``. For non-HF tokenizers, pass ``None``
            and call the bridge with token-id tensors directly.
        dtype: Optional dtype to record on ``cfg.dtype``. If ``None``, the
            dtype is read from the model's first parameter — passing
            explicit ``torch.float32`` when the model is actually bf16
            would write a stale value to the config that downstream code
            (weight processing, interventions) would read incorrectly.
        device: Informational only. If supplied, recorded on
            ``cfg.device``. Otherwise inferred from the model's first
            parameter.
        model_name: Recorded on ``cfg.model_name``. Defaults to ``"external"``.

    Returns:
        A :class:`TransformerBridge` wrapping the supplied model.

    Raises:
        ValueError: If neither ``hf_config`` nor ``tl_config`` is supplied,
            if both are supplied (ambiguous — pick one), or if
            ``architecture`` is not in :data:`SUPPORTED_ARCHITECTURES`.
    """
    if hf_config is None and tl_config is None:
        raise ValueError(
            "boot_external requires exactly one of hf_config or tl_config — "
            "the bridge needs config fields (d_model, n_heads, n_layers, ...) "
            "that can't be inferred from the model alone."
        )
    if hf_config is not None and tl_config is not None:
        raise ValueError(
            "boot_external got both hf_config and tl_config; supply exactly one. "
            "hf_config triggers HF→bridge translation; tl_config bypasses it."
        )
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Supported: {sorted(SUPPORTED_ARCHITECTURES.keys())}"
        )

    # Resolve dtype from the model itself unless caller forces it. Defaulting
    # to fp32 here would silently lie to downstream code about a bf16 model.
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    if tl_config is not None:
        bridge_config = tl_config
        bridge_config.architecture = architecture
        # Explicit kwarg always wins; otherwise keep whatever tl_config carries;
        # otherwise stamp our default. No magic sentinel comparison.
        if model_name != "external" or not getattr(bridge_config, "model_name", None):
            bridge_config.model_name = model_name
        bridge_config.dtype = dtype
    else:
        bridge_config = build_bridge_config_from_hf(
            hf_config, architecture, model_name, dtype
        )

    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    if device is not None:
        adapter.cfg.device = str(device)
    else:
        try:
            adapter.cfg.device = str(next(model.parameters()).device)
        except StopIteration:
            adapter.cfg.device = "cpu"

    adapter.prepare_model(model)

    if tokenizer is not None:
        default_padding_side = getattr(adapter.cfg, "default_padding_side", None)
        tokenizer = setup_tokenizer(tokenizer, default_padding_side=default_padding_side)
        adapter.cfg.tokenizer_prepends_bos, adapter.cfg.tokenizer_appends_eos = (
            detect_tokenizer_bos_eos(tokenizer)
        )

    return TransformerBridge(model, adapter, tokenizer)


setattr(TransformerBridge, "boot_external", staticmethod(boot_external))
