"""Weight init for NativeModel.

Supported modes: ``"gpt2"`` (Normal(0, std) with 1/sqrt(2*n_layers) residual
scaling on output projections), ``"xavier_uniform"`` / ``"xavier_normal"``,
``"kaiming_uniform"`` / ``"kaiming_normal"`` (relu nonlinearity). Norm weights
go to 1, all biases to 0.

Determinism uses a scoped ``torch.Generator``, not ``torch.manual_seed``, so
seeded init does not perturb the caller's global RNG.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, cast

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

# Residual-scaled output is gpt2-specific; other modes treat every weight the
# same. Each entry takes ``(tensor, generator)`` to thread the scoped Generator.
_NonResidualInit = Callable[[torch.Tensor, Optional[torch.Generator]], torch.Tensor]
_NON_RESIDUAL_MODES: dict[str, _NonResidualInit] = {
    "xavier_uniform": lambda t, g: nn.init.xavier_uniform_(t, generator=g),
    "xavier_normal": lambda t, g: nn.init.xavier_normal_(t, generator=g),
    "kaiming_uniform": lambda t, g: nn.init.kaiming_uniform_(t, nonlinearity="relu", generator=g),
    "kaiming_normal": lambda t, g: nn.init.kaiming_normal_(t, nonlinearity="relu", generator=g),
}

_SUPPORTED_MODES = frozenset({"gpt2", *_NON_RESIDUAL_MODES})


def initialize_native_model(
    model: NativeModel, cfg: TransformerBridgeConfig, seed: int | None = None
) -> None:
    """Initialize ``model`` weights in-place. Honors ``cfg.init_mode`` and ``cfg.seed``."""
    effective_seed = seed if seed is not None else cfg.seed

    # Scoped generator on the model's device — None falls back to the global RNG.
    try:
        gen_device = next(model.parameters()).device
    except StopIteration:
        gen_device = torch.device("cpu")
    generator: Optional[torch.Generator]
    if effective_seed is not None:
        g = torch.Generator(device=gen_device)
        g.manual_seed(effective_seed)
        generator = g
    else:
        generator = None

    init_mode = (cfg.init_mode or "gpt2").lower()
    if init_mode not in _SUPPORTED_MODES:
        raise NotImplementedError(
            f"init_mode={init_mode!r} is not supported for NativeModel. "
            f"Supported modes: {sorted(_SUPPORTED_MODES)}."
        )

    weight_init: Callable[[torch.Tensor], torch.Tensor]
    output_init: Callable[[torch.Tensor], torch.Tensor]
    if init_mode == "gpt2":
        std = cfg.initializer_range if cfg.initializer_range > 0 else 0.02
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        weight_init = lambda t: nn.init.normal_(
            t, mean=0.0, std=std, generator=generator
        )  # noqa: E731
        output_init = lambda t: nn.init.normal_(  # noqa: E731
            t, mean=0.0, std=std * residual_scale, generator=generator
        )
    else:
        fn = _NON_RESIDUAL_MODES[init_mode]
        weight_init = lambda t: fn(t, generator)  # noqa: E731
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
    elif isinstance(norm, nn.Identity):
        pass
    else:
        raise TypeError(f"Unknown normalization type: {type(norm).__name__}")


def _init_block(
    block: NativeBlock,
    *,
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
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
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
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
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    weight_init(mlp.fc_in.weight)
    nn.init.zeros_(mlp.fc_in.bias)
    output_init(mlp.fc_out.weight)
    nn.init.zeros_(mlp.fc_out.bias)


def _init_gated_mlp(
    mlp: NativeGatedMLP,
    *,
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    weight_init(mlp.gate.weight)
    # ``in`` is registered via add_module; getattr resolves it from _modules.
    in_proj = cast(nn.Linear, getattr(mlp, "in"))
    weight_init(in_proj.weight)
    output_init(mlp.out.weight)
