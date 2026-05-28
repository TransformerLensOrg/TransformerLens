"""Canonical hook names ↔ (layer, kind) for the Inspect HF provider.

Torch-free; shared by the provider (what to capture / where to intervene) and the
driver (supported set, decode). Covers only the ``d_model``-shaped boundaries our
HF forward hooks match ``boot_transformers`` on exactly. Head-split hooks
(q/k/v/z, pattern), ``embed.hook_out`` (inline for some architectures), and
``ln_final.hook_normalized`` (fold-LN convention) are intentionally excluded —
they're reported non-fireable, matching the vLLM driver.
"""
from __future__ import annotations

import re

# kind → the canonical TL hook name template(s) that read that boundary. The two
# attn/mlp aliases are the same tensor under TL's two naming conventions.
_KIND_NAMES: dict[str, list[str]] = {
    "resid_pre": ["blocks.{i}.hook_resid_pre"],
    "resid_mid": ["blocks.{i}.hook_resid_mid"],
    "resid_post": ["blocks.{i}.hook_resid_post"],
    "attn_out": ["blocks.{i}.hook_attn_out", "blocks.{i}.attn.hook_out"],
    "mlp_out": ["blocks.{i}.hook_mlp_out", "blocks.{i}.mlp.hook_out"],
}

# Kinds reachable by a forward hook on a clean module boundary. resid_mid is
# derived (resid_post − mlp_out), so it's capture-only — interventions can't target it.
INTERVENEABLE_KINDS = frozenset({"resid_pre", "attn_out", "mlp_out", "resid_post"})

_SUFFIX_TO_KIND = {
    "hook_resid_pre": "resid_pre",
    "hook_resid_mid": "resid_mid",
    "hook_resid_post": "resid_post",
    "hook_attn_out": "attn_out",
    "attn.hook_out": "attn_out",
    "hook_mlp_out": "mlp_out",
    "mlp.hook_out": "mlp_out",
}
_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")


def supported_hook_points(n_layers: int) -> frozenset[str]:
    """Every fireable hook name across all layers."""
    return frozenset(
        name.format(i=i)
        for i in range(n_layers)
        for names in _KIND_NAMES.values()
        for name in names
    )


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
