"""Canonical hook names ↔ (layer, kind) for the Inspect HF provider.

Torch-free; shared by the provider (capture/intervene) and the driver (supported set,
decode). Covers the ``d_model``-shaped decoder-layer boundaries; head-split hooks
(q/k/v/z, pattern), ``embed``, and ``ln_final`` (fold-LN convention) are non-fireable.

Names are TransformerBridge-native (``blocks.{i}.hook_out``, ``.attn.hook_out``, ...),
matching the vLLM driver — not the HookedTransformer aliases. A bridge cache carries
both with identical values, so parity vs ``boot_transformers`` still resolves.

Parity is verified only for ``PARITY_VERIFIED_ARCHITECTURES``; otherwise the boundaries
assume the standard HF decoder-layer layout (``boot_inspect`` warns).
"""
from __future__ import annotations

import re

# Architectures whose boundary↔name mapping is checked against boot_transformers
# in tests/acceptance/model_bridge/test_inspect_provider.py.
PARITY_VERIFIED_ARCHITECTURES = frozenset({"GPT2LMHeadModel"})

# One canonical TransformerBridge name per boundary (no aliases — avoids duplicate
# HookPoints/cache entries). resid_mid (ln2.hook_in) is derived (resid_pre +
# attn_out), so it's capture-only.
_KIND_NAMES = {
    "resid_pre": "blocks.{i}.hook_in",
    "resid_mid": "blocks.{i}.ln2.hook_in",
    "resid_post": "blocks.{i}.hook_out",
    "attn_out": "blocks.{i}.attn.hook_out",
    "mlp_out": "blocks.{i}.mlp.hook_out",
}
INTERVENEABLE_KINDS = frozenset({"resid_pre", "attn_out", "mlp_out", "resid_post"})

_SUFFIX_TO_KIND = {
    "hook_in": "resid_pre",
    "ln2.hook_in": "resid_mid",
    "hook_out": "resid_post",
    "attn.hook_out": "attn_out",
    "mlp.hook_out": "mlp_out",
}
_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")


def supported_hook_points(n_layers: int) -> frozenset[str]:
    """Every fireable hook name across all layers (one per boundary)."""
    return frozenset(name.format(i=i) for i in range(n_layers) for name in _KIND_NAMES.values())


def nonfireable_hook_points(n_layers: int) -> frozenset[str]:
    """Hooks the residual/attn/mlp provider can't fire (head-split, embed, ln_final)."""
    names = ["embed.hook_out", "ln_final.hook_normalized", "unembed.hook_out"]
    for i in range(n_layers):
        names += [
            f"blocks.{i}.attn.hook_pattern",
            f"blocks.{i}.attn.hook_attn_scores",
            f"blocks.{i}.attn.hook_q",
            f"blocks.{i}.attn.hook_k",
            f"blocks.{i}.attn.hook_v",
            f"blocks.{i}.attn.hook_z",
        ]
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
