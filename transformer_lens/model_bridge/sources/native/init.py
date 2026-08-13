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
    NativeLayerNormPre,
    NativeMLP,
    NativeModel,
    NativeRMSNorm,
    NativeRMSNormPre,
)

# Residual-scaled output is gpt2-specific; other modes treat every weight the
# same. Each entry takes ``(tensor, generator, gain)`` — gain honors
# ``cfg.initializer_range`` like the legacy init did (which passed it as the
# xavier/kaiming gain); kaiming has no gain kwarg, so scale after.
_NonResidualInit = Callable[[torch.Tensor, Optional[torch.Generator], float], torch.Tensor]
_NON_RESIDUAL_MODES: dict[str, _NonResidualInit] = {
    "xavier_uniform": lambda t, g, gain: nn.init.xavier_uniform_(t, gain=gain, generator=g),
    "xavier_normal": lambda t, g, gain: nn.init.xavier_normal_(t, gain=gain, generator=g),
    "kaiming_uniform": lambda t, g, gain: nn.init.kaiming_uniform_(
        t, nonlinearity="relu", generator=g
    ).mul_(gain),
    "kaiming_normal": lambda t, g, gain: nn.init.kaiming_normal_(
        t, nonlinearity="relu", generator=g
    ).mul_(gain),
}

_SUPPORTED_MODES = frozenset({"gpt2", *_NON_RESIDUAL_MODES})


def _unwrap_component(module: nn.Module) -> nn.Module:
    """Return the native module stored behind a bridge wrapper, if present."""
    original = getattr(module, "original_component", None)
    return original if isinstance(original, nn.Module) else module


def initialize_native_model(
    model: NativeModel, cfg: TransformerBridgeConfig, seed: int | None = None
) -> None:
    """Initialize ``model`` weights in-place. Honors ``cfg.init_mode`` and ``cfg.seed``."""
    effective_seed = seed if seed is not None else cfg.seed

    # Always generate on CPU/fp32 and copy into the parameter: boot initializes
    # before .to(device)/.to(dtype) while init_weights() runs after, and a
    # generator seeded on the live parameter device produces a different stream
    # — the same seed must reproduce the same weights either way.
    generator: Optional[torch.Generator]
    if effective_seed is not None:
        g = torch.Generator()
        g.manual_seed(effective_seed)
        generator = g
    else:
        generator = None

    def _staged(
        fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def apply(t: torch.Tensor) -> torch.Tensor:
            staging = torch.empty(t.shape, dtype=torch.float32)
            fn(staging)
            with torch.no_grad():
                t.copy_(staging)
            return t

        return apply

    init_mode = (cfg.init_mode or "gpt2").lower()
    if init_mode not in _SUPPORTED_MODES:
        raise NotImplementedError(
            f"init_mode={init_mode!r} is not supported for NativeModel. "
            f"Supported modes: {sorted(_SUPPORTED_MODES)}."
        )

    weight_init: Callable[[torch.Tensor], torch.Tensor]
    output_init: Callable[[torch.Tensor], torch.Tensor]
    if init_mode == "gpt2":
        # Default matches the legacy TL scheme: N(0, 0.64/d_model), i.e.
        # std = 0.8/sqrt(d_model), not GPT-2's paper 0.02 — toy-model training
        # dynamics (e.g. the grokking demo) depend on this scale.
        std = cfg.initializer_range if cfg.initializer_range > 0 else 0.8 / math.sqrt(cfg.d_model)
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        weight_init = lambda t: nn.init.normal_(
            t, mean=0.0, std=std, generator=generator
        )  # noqa: E731
        output_init = lambda t: nn.init.normal_(  # noqa: E731
            t, mean=0.0, std=std * residual_scale, generator=generator
        )
    else:
        fn = _NON_RESIDUAL_MODES[init_mode]
        # Honor an explicitly-set initializer_range as the gain (legacy
        # behavior); the sentinel/default keeps plain xavier/kaiming scaling.
        gain = cfg.initializer_range if cfg.initializer_range > 0 else 1.0
        weight_init = lambda t: fn(t, generator, gain)  # noqa: E731
        output_init = weight_init

    weight_init = _staged(weight_init)
    output_init = _staged(output_init)

    tok_embed = cast(nn.Embedding, _unwrap_component(model.tok_embed))
    weight_init(tok_embed.weight)
    if model.pos is not None:
        pos = cast(nn.Embedding, _unwrap_component(model.pos))
        weight_init(pos.weight)
    # Rotary has only registered buffers (cos/sin), no parameters to init.

    for block in model.layers:
        native_block = cast(NativeBlock, _unwrap_component(block))
        _init_block(native_block, weight_init=weight_init, output_init=output_init)

    _init_norm(model.ln_out)
    head = cast(nn.Linear, _unwrap_component(model.head))
    weight_init(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def _init_norm(norm: nn.Module) -> None:
    norm = _unwrap_component(norm)
    if isinstance(norm, NativeRMSNorm):
        nn.init.ones_(norm.weight)
    elif isinstance(norm, (NativeRMSNormPre, NativeLayerNormPre)):
        pass
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
    attn = cast(NativeAttention, _unwrap_component(block.attn))
    _init_attention(attn, weight_init=weight_init, output_init=output_init)
    if not block.cfg.attn_only:
        _init_norm(block.ln2)
        mlp = _unwrap_component(block.mlp)
        if isinstance(mlp, NativeGatedMLP):
            _init_gated_mlp(mlp, weight_init=weight_init, output_init=output_init)
        else:
            _init_mlp(
                cast(NativeMLP, mlp),
                weight_init=weight_init,
                output_init=output_init,
            )


def _init_attention(
    attn: NativeAttention,
    *,
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    for component in (attn.q, attn.k, attn.v):
        linear = cast(nn.Linear, _unwrap_component(component))
        weight_init(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    output = cast(nn.Linear, _unwrap_component(attn.o))
    output_init(output.weight)
    if output.bias is not None:
        nn.init.zeros_(output.bias)


def _init_mlp(
    mlp: NativeMLP,
    *,
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    fc_in = cast(nn.Linear, _unwrap_component(mlp.fc_in))
    fc_out = cast(nn.Linear, _unwrap_component(mlp.fc_out))
    weight_init(fc_in.weight)
    nn.init.zeros_(fc_in.bias)
    output_init(fc_out.weight)
    nn.init.zeros_(fc_out.bias)


def _init_gated_mlp(
    mlp: NativeGatedMLP,
    *,
    weight_init: Callable[[torch.Tensor], torch.Tensor],
    output_init: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    gate = cast(nn.Linear, _unwrap_component(mlp.gate))
    weight_init(gate.weight)
    # ``in`` is registered via add_module; getattr resolves it from _modules.
    in_proj = cast(nn.Linear, _unwrap_component(getattr(mlp, "in")))
    out_proj = cast(nn.Linear, _unwrap_component(mlp.out))
    weight_init(in_proj.weight)
    output_init(out_proj.weight)
