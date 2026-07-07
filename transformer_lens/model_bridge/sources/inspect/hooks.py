"""Canonical hook names ↔ (layer, kind) for the Inspect HF provider.

Torch-free; shared by the provider (capture/intervene) and the driver (supported set,
decode). Covers the ``d_model``-shaped decoder-layer boundaries plus the head-split
attention hooks (q/k/v/z, pattern) where the provider's structural probe finds the
projections; ``attn_scores``, ``embed``, and ``ln_final`` (fold-LN convention) stay
non-fireable.

Names are TransformerBridge-native (``blocks.{i}.hook_out``, ``.attn.hook_out``, ...),
not the HookedTransformer aliases. A bridge cache carries both with identical values,
so parity vs ``boot_transformers`` still resolves.

Which boundaries are actually fireable is decided per-model by the provider's structural
self-check, not a hand-kept architecture list: it locates attn/mlp and probes whether the
``resid_pre + attn_out`` derivation holds, gating ``resid_mid`` otherwise; head-split
kinds need separate q/k/v projections (q/k/v), a locatable out-projection (z), or eager
attention (pattern). ``supported_hook_points(n_layers, kinds=...)`` filters to that set.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

ALL_KINDS = frozenset({"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"})
# Head-split attention kinds, served only when structurally detected (never by default):
# q/k/v are the pre-RoPE projection outputs, z is the out-projection input (all
# ``(seq, heads, d_head)``); pattern is post-softmax attention ``(heads, q_pos, k_pos)``.
HEAD_KINDS = frozenset({"q", "k", "v", "z", "pattern"})

# One canonical TransformerBridge name per boundary (no aliases — avoids duplicate
# HookPoints/cache entries). resid_mid (ln2.hook_in) is derived (resid_pre +
# attn_out), so it's capture-only.
_KIND_NAMES = {
    "resid_pre": "blocks.{i}.hook_in",
    "resid_mid": "blocks.{i}.ln2.hook_in",
    "resid_post": "blocks.{i}.hook_out",
    "attn_out": "blocks.{i}.attn.hook_out",
    "mlp_out": "blocks.{i}.mlp.hook_out",
    "q": "blocks.{i}.attn.hook_q",
    "k": "blocks.{i}.attn.hook_k",
    "v": "blocks.{i}.attn.hook_v",
    "z": "blocks.{i}.attn.hook_z",
    "pattern": "blocks.{i}.attn.hook_pattern",
}
# pattern is read from the forward's output_attentions — nothing to write back, so it's
# capture-only (like the derived resid_mid).
INTERVENEABLE_KINDS = frozenset(
    {"resid_pre", "attn_out", "mlp_out", "resid_post", "q", "k", "v", "z"}
)

# Rank of each kind's batchless wire array; the driver unsqueezes exactly one batch dim.
WIRE_BATCHLESS_NDIM = {
    **{kind: 2 for kind in ALL_KINDS},  # (seq, d_model)
    "q": 3,
    "k": 3,
    "v": 3,
    "z": 3,  # (seq, heads, d_head)
    "pattern": 3,  # (heads, q_pos, k_pos)
}

_SUFFIX_TO_KIND = {
    "hook_in": "resid_pre",
    "ln2.hook_in": "resid_mid",
    "hook_out": "resid_post",
    "attn.hook_out": "attn_out",
    "mlp.hook_out": "mlp_out",
    "attn.hook_q": "q",
    "attn.hook_k": "k",
    "attn.hook_v": "v",
    "attn.hook_z": "z",
    "attn.hook_pattern": "pattern",
}
_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")


def supported_hook_points(n_layers: int, kinds: Optional[Iterable[str]] = None) -> frozenset[str]:
    """Fireable hook names across all layers. ``kinds=None`` means all *boundary* kinds
    (head-split kinds are opt-in — a provider must detect and list them explicitly);
    pass the provider's detected kinds to gate (e.g. drop ``resid_mid`` for parallel)."""
    selected = ALL_KINDS if kinds is None else kinds
    return frozenset(_KIND_NAMES[k].format(i=i) for i in range(n_layers) for k in selected)


def all_hook_points(n_layers: int) -> frozenset[str]:
    """Every hook name the registry can serve (boundaries + head-split) — the universe a
    driver subtracts its supported set from to build ``non_fireable_hook_points``."""
    return supported_hook_points(n_layers, ALL_KINDS | HEAD_KINDS)


def nonfireable_hook_points(n_layers: int) -> frozenset[str]:
    """Hooks no provider configuration can fire (embed, ln_final, pre-softmax scores).
    Head-split q/k/v/z/pattern are *conditionally* fireable and belong here only when a
    model's structural probe gates them — the driver handles that subtraction."""
    names = ["embed.hook_out", "ln_final.hook_normalized", "unembed.hook_out"]
    names += [f"blocks.{i}.attn.hook_attn_scores" for i in range(n_layers)]
    return frozenset(names)


def resolve(name: str) -> tuple[int, str] | None:
    """Canonical hook name → ``(layer, kind)``, or ``None`` if not a fireable hook."""
    match = _BLOCK.match(name)
    if match is None:
        return None
    kind = _SUFFIX_TO_KIND.get(match.group(2))
    return (int(match.group(1)), kind) if kind is not None else None


def wire_key(layer: int, kind: str) -> str:
    """Stable key for one captured boundary in the activation payload."""
    return f"{layer}:{kind}"


def name_from_wire_key(key: str) -> str | None:
    """Inverse of ``wire_key`` ∘ ``resolve``: ``"<layer>:<kind>"`` → the canonical hook
    name, or ``None`` if the kind is unknown."""
    layer, _, kind = key.partition(":")
    template = _KIND_NAMES.get(kind)
    return template.format(i=int(layer)) if template and layer.isdigit() else None
