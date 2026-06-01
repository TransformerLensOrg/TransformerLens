"""Weight init for NativeModel. Supports the standard TL init modes.

Modes:

- ``"gpt2"`` (default): Normal(0, initializer_range) for everything, with
  ``1/sqrt(2 * n_layers)`` residual scaling on attn.o and mlp output. Mirrors
  HookedTransformer's ``init_mode='gpt2'`` and is the right default for GPT-style
  toy models.
- ``"xavier_uniform"`` / ``"xavier_normal"``: ``torch.nn.init.xavier_{uniform,normal}_``
  on linear weights and embeddings. Useful for sanity-checking sensitivity to
  init scheme without changing architecture.
- ``"kaiming_uniform"`` / ``"kaiming_normal"``: ``torch.nn.init.kaiming_{uniform,normal}_``
  with ``nonlinearity='relu'``. Reasonable default for ReLU-family activations.

For every mode: LayerNorm/RMSNorm weights init to 1, biases to 0; Linear biases
init to 0.

HookedTransformer's exact init pulls from a private RNG. We rely on
``torch.manual_seed`` plus per-layer ``torch.nn.init`` for reproducibility.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig

from .model import (
    NativeAttention,
    NativeBlock,
    NativeGatedMLP,
    NativeMLP,
    NativeModel,
    NativeRMSNorm,
)

# Init modes that don't depend on residual depth — they treat every weight
# identically. The residual-scaled output projection trick is gpt2-specific.
_NON_RESIDUAL_MODES: dict[str, Callable[[torch.Tensor], None]] = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": lambda t: nn.init.kaiming_uniform_(t, nonlinearity="relu"),
    "kaiming_normal": lambda t: nn.init.kaiming_normal_(t, nonlinearity="relu"),
}

_SUPPORTED_MODES = frozenset({"gpt2", *_NON_RESIDUAL_MODES})


def initialize_native_model(
    model: NativeModel, cfg: TransformerBridgeConfig, seed: int | None = None
) -> None:
    """Initialize ``model`` weights in-place. Honors ``cfg.init_mode`` and ``cfg.seed``."""
    effective_seed = seed if seed is not None else cfg.seed
    if effective_seed is not None:
        torch.manual_seed(effective_seed)

    init_mode = (cfg.init_mode or "gpt2").lower()
    if init_mode not in _SUPPORTED_MODES:
        raise NotImplementedError(
            f"init_mode={init_mode!r} is not supported for NativeModel. "
            f"Supported modes: {sorted(_SUPPORTED_MODES)}."
        )

    if init_mode == "gpt2":
        std = cfg.initializer_range if cfg.initializer_range > 0 else 0.02
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        weight_init = lambda t: nn.init.normal_(t, mean=0.0, std=std)  # noqa: E731
        output_init = lambda t: nn.init.normal_(t, mean=0.0, std=std * residual_scale)  # noqa: E731
    else:
        # Non-gpt2 modes ignore residual-depth scaling — they have their own
        # gain rules that already account for the layer's role.
        weight_init = _NON_RESIDUAL_MODES[init_mode]
        output_init = weight_init

    weight_init(model.tok_embed.weight)
    if model.pos is not None:
        weight_init(model.pos.weight)
    # Rotary has only registered buffers (cos/sin), no parameters to init.

    for block in model.layers:
        _init_block(block, weight_init=weight_init, output_init=output_init)

    _init_norm(model.ln_out)
    weight_init(model.head.weight)


def _init_norm(norm: nn.Module) -> None:
    if isinstance(norm, NativeRMSNorm):
        nn.init.ones_(norm.weight)
    elif isinstance(norm, nn.LayerNorm):
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)
    else:
        raise TypeError(f"Unknown normalization type: {type(norm).__name__}")


def _init_block(
    block: NativeBlock,
    *,
    weight_init: Callable[[torch.Tensor], None],
    output_init: Callable[[torch.Tensor], None],
) -> None:
    _init_norm(block.ln1)
    _init_attention(block.attn, weight_init=weight_init, output_init=output_init)
    if not block.cfg.attn_only:
        _init_norm(block.ln2)
        if isinstance(block.mlp, NativeGatedMLP):
            _init_gated_mlp(block.mlp, weight_init=weight_init, output_init=output_init)
        else:
            _init_mlp(block.mlp, weight_init=weight_init, output_init=output_init)


def _init_attention(
    attn: NativeAttention,
    *,
    weight_init: Callable[[torch.Tensor], None],
    output_init: Callable[[torch.Tensor], None],
) -> None:
    for linear in (attn.q, attn.k, attn.v):
        weight_init(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    output_init(attn.o.weight)
    if attn.o.bias is not None:
        nn.init.zeros_(attn.o.bias)


def _init_mlp(
    mlp: NativeMLP,
    *,
    weight_init: Callable[[torch.Tensor], None],
    output_init: Callable[[torch.Tensor], None],
) -> None:
    weight_init(mlp.fc_in.weight)
    nn.init.zeros_(mlp.fc_in.bias)
    output_init(mlp.fc_out.weight)
    nn.init.zeros_(mlp.fc_out.bias)


def _init_gated_mlp(
    mlp: NativeGatedMLP,
    *,
    weight_init: Callable[[torch.Tensor], None],
    output_init: Callable[[torch.Tensor], None],
) -> None:
    weight_init(mlp.gate.weight)
    # ``in`` is registered via add_module; getattr resolves it from _modules.
    weight_init(getattr(mlp, "in").weight)
    output_init(mlp.out.weight)
