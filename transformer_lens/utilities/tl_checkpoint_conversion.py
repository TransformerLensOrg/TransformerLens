"""One-time converter for legacy TL-property-format checkpoints (#1588).

Historical training runs (OthelloGPT, grokking demos, ARENA content) were
saved via ``HookedTransformer.state_dict()`` before ``TransformerBridge``
existed, using property-style keys ("blocks.0.attn.W_Q", "embed.W_E", ...)
and per-head tensor shapes. ``convert_tl_checkpoint`` maps those onto the
key/tensor format ``TransformerBridge.boot_native(cfg).load_state_dict``
accepts natively, so these checkpoints can be converted once and re-saved in
bridge format. This is deliberately a standalone converter rather than a
second key convention taught to ``load_state_dict`` itself: convert once,
``bridge.load_state_dict(converted)``, then re-save with ``bridge.state_dict()``.
"""

from __future__ import annotations

from typing import Callable, Optional

import einops
import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig

# Buffers that live on HookedTransformer's attention blocks but have no
# Parameter counterpart on the bridge side (causal mask, IGNORE sentinel).
_DROPPED_BUFFER_SUFFIXES = (".mask", ".IGNORE")

TensorConvert = Callable[[torch.Tensor, TransformerBridgeConfig, str], torch.Tensor]


def _validate_shape(tensor: torch.Tensor, expected: tuple[int, ...], key: str) -> None:
    # Merging/splitting per-head dims (unlike a plain transpose) produces a
    # validly-shaped result for *any* head count, since d_model == n_heads *
    # d_head for any factoring of it — a wrong cfg silently mis-groups heads
    # without ever tripping a downstream shape-mismatch error. Check the
    # untouched per-head shape explicitly before reshaping.
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"convert_tl_checkpoint: {key!r} has shape {tuple(tensor.shape)}, "
            f"expected {expected} for the given cfg. The checkpoint may not "
            "match this cfg (n_heads/n_key_value_heads/d_head/d_model)."
        )


def _kv_heads(cfg: TransformerBridgeConfig) -> int:
    return cfg.n_key_value_heads or cfg.n_heads


def _convert_w_q(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    _validate_shape(t, (cfg.n_heads, cfg.d_model, cfg.d_head), key)
    return einops.rearrange(t, "n_heads d_model d_head -> (n_heads d_head) d_model")


def _convert_w_kv(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    _validate_shape(t, (_kv_heads(cfg), cfg.d_model, cfg.d_head), key)
    return einops.rearrange(t, "n_heads d_model d_head -> (n_heads d_head) d_model")


def _convert_w_o(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    _validate_shape(t, (cfg.n_heads, cfg.d_head, cfg.d_model), key)
    return einops.rearrange(t, "n_heads d_head d_model -> d_model (n_heads d_head)")


def _convert_b_q(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    _validate_shape(t, (cfg.n_heads, cfg.d_head), key)
    return einops.rearrange(t, "n_heads d_head -> (n_heads d_head)")


def _convert_b_kv(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    _validate_shape(t, (_kv_heads(cfg), cfg.d_head), key)
    return einops.rearrange(t, "n_heads d_head -> (n_heads d_head)")


def _identity(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    return t


def _transpose(t: torch.Tensor, cfg: TransformerBridgeConfig, key: str) -> torch.Tensor:
    return t.T.contiguous()


# Old TL-property suffix -> (new bridge-key suffix, tensor conversion).
# Checked in order, longest/most-specific first, so e.g. ".b_Q" is matched
# before the generic ".b" LayerNorm-bias fallback.
_SUFFIX_CONVERSIONS: list[tuple[str, str, TensorConvert]] = [
    (".W_Q", ".q.weight", _convert_w_q),
    # GQA stores K/V under a leading-underscore name (the raw Parameter);
    # plain ".W_K"/".W_V" become expanding (non-Parameter) properties instead.
    ("._W_K", ".k.weight", _convert_w_kv),
    ("._W_V", ".v.weight", _convert_w_kv),
    (".W_K", ".k.weight", _convert_w_kv),
    (".W_V", ".v.weight", _convert_w_kv),
    (".W_O", ".o.weight", _convert_w_o),
    (".b_Q", ".q.bias", _convert_b_q),
    ("._b_K", ".k.bias", _convert_b_kv),
    ("._b_V", ".v.bias", _convert_b_kv),
    (".b_K", ".k.bias", _convert_b_kv),
    (".b_V", ".v.bias", _convert_b_kv),
    (".b_O", ".o.bias", _identity),
    (".W_in", ".in.weight", _transpose),
    (".b_in", ".in.bias", _identity),
    (".W_out", ".out.weight", _transpose),
    (".b_out", ".out.bias", _identity),
    (".W_gate", ".gate.weight", _transpose),
    (".b_gate", ".gate.bias", _identity),
    (".W_U", ".weight", _transpose),
    (".b_U", ".bias", _identity),
    (".W_E", ".weight", _identity),
    (".W_pos", ".weight", _identity),
    (".w", ".weight", _identity),
    (".b", ".bias", _identity),
]


def _convert_key_and_tensor(
    key: str, tensor: torch.Tensor, cfg: TransformerBridgeConfig
) -> Optional[tuple[str, torch.Tensor]]:
    for old_suffix, new_suffix, convert in _SUFFIX_CONVERSIONS:
        if key.endswith(old_suffix):
            new_key = key[: -len(old_suffix)] + new_suffix
            return new_key, convert(tensor, cfg, key)
    return None


def convert_tl_checkpoint(
    state_dict: dict[str, torch.Tensor],
    cfg: TransformerBridgeConfig,
) -> dict[str, torch.Tensor]:
    """Convert a legacy TL-property-format state dict to the key/tensor
    format ``TransformerBridge.boot_native(cfg).load_state_dict`` accepts.

    Args:
        state_dict: A state dict in the old ``HookedTransformer`` convention
            (e.g. from ``HookedTransformer.state_dict()``), with keys like
            ``"blocks.0.attn.W_Q"`` and per-head tensor shapes.
        cfg: The config the checkpoint was trained/saved under. Used both to
            reshape per-head attention weights and to validate that the
            checkpoint's per-head shapes actually match this cfg — a
            mismatched cfg would otherwise silently mis-group heads without
            ever tripping a shape error, since d_model == n_heads * d_head
            holds for any wrong factoring too.

    Returns:
        A state dict with modern bridge keys (e.g. ``"blocks.0.attn.q.weight"``)
        and flat ``nn.Linear``-oriented tensor shapes, ready for
        ``bridge.load_state_dict(converted, strict=True)``.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(_DROPPED_BUFFER_SUFFIXES):
            continue
        result = _convert_key_and_tensor(key, tensor, cfg)
        if result is None:
            raise ValueError(
                f"convert_tl_checkpoint: don't know how to convert key {key!r} "
                "(not a recognized TL-property parameter or buffer suffix)."
            )
        new_key, new_tensor = result
        converted[new_key] = new_tensor
    return converted
