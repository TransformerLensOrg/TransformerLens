"""TL-native model source for TransformerBridge."""
from __future__ import annotations

import copy as _copy
from typing import Any, Optional, Union

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import build_bridge_from_module
from transformer_lens.model_bridge.sources.native.init import initialize_native_model
from transformer_lens.model_bridge.sources.native.model import (
    NativeAttention,
    NativeBlock,
    NativeMLP,
    NativeModel,
)


def boot(
    config: Union[TransformerBridgeConfig, dict],
    tokenizer: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    model_name: str = "native",
) -> TransformerBridge:
    """Build a bridge around a small, randomly-initialized TL-native model.

    No HuggingFace Hub call, no ``transformers`` import. ``config.init_mode``
    and ``config.seed`` control reproducibility.
    """
    cfg: TransformerBridgeConfig
    if isinstance(config, dict):
        cfg = TransformerBridgeConfig.from_dict(config)
    else:
        # Deep-copy so NativeModel's default-resolution writes don't land
        # on the caller's config.
        cfg = _copy.deepcopy(config)

    # Foreign architecture strings would dispatch to the wrong adapter and
    # crash deep in prepare_model. Refuse them with a pointing message.
    if cfg.architecture not in (None, "TransformerLensNative"):
        raise ValueError(
            f"boot_native cannot build a {cfg.architecture!r} model — "
            f"it only constructs the TL-native architecture. Either clear "
            f"config.architecture or set it to 'TransformerLensNative', "
            f"or use boot_transformers / build_bridge_from_module for "
            f"non-native architectures."
        )
    architecture = "TransformerLensNative"

    # Fork RNG around construction + init when seeded so neither nn.Linear's
    # default reset_parameters nor our scoped init perturb the caller's RNG.
    # Unseeded calls let global RNG advance normally.
    if cfg.seed is not None:
        with torch.random.fork_rng(devices=[]):
            model = NativeModel(cfg)
            initialize_native_model(model, cfg)
    else:
        model = NativeModel(cfg)
        initialize_native_model(model, cfg)

    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)

    return build_bridge_from_module(
        model,
        architecture=architecture,
        tl_config=cfg,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        model_name=model_name,
    )


# Attach to TransformerBridge as a staticmethod, matching boot_transformers / boot_vllm.
setattr(TransformerBridge, "boot_native", staticmethod(boot))


__all__ = [
    "NativeAttention",
    "NativeBlock",
    "NativeMLP",
    "NativeModel",
    "boot",
    "initialize_native_model",
]
