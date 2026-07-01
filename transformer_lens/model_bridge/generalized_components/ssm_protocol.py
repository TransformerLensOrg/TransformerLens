"""Family-agnostic discovery + shared state-mutation surface for SSM/recurrent mixers.

Defines the structural contract (``SSMMixerProtocol``) that ``SSM2MixerBridge``
(Mamba-2), ``SSMMixerBridge`` (Mamba-1), and ``GatedDeltaNetBridge`` all satisfy,
plus a lookup that finds a block's SSM mixer regardless of which variant slot it
occupies (``.mixer`` for Mamba, ``.linear_attn`` for gated-delta-net). A Protocol
(not a base class) keeps each family's own hook set — only the discovery surface
is shared. ``SSMStateHookMixin`` is the one place the canonical eager-scan
intervention hooks (``hook_ssm_state`` + the ``eager_scan`` opt-in) are defined, so
the three families share a single definition rather than repeating it ad hoc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.block import (
    VARIANT_SUBMODULE_NAMES,
)

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


@runtime_checkable
class SSMMixerProtocol(Protocol):
    """An SSM/recurrent mixer that can materialize an effective-attention matrix.

    Only ``compute_effective_attention(cache, layer_idx)`` is required; families
    add their own optional keyword options (e.g. ``include_dt_scaling``,
    ``per_state_coord``) which callers discover by signature introspection.
    """

    def compute_effective_attention(self, cache: "ActivationCache", layer_idx: int) -> torch.Tensor:
        ...


class SSMStateHookMixin:
    """Canonical state-mutation hooks shared by the SSM/recurrent mixers.

    One definition point so Mamba-1/Mamba-2/gated-delta-net expose the same names:

    - ``hook_ssm_state`` — post-scan state trajectory ``S_t`` (created here for all);
      patching it changes only the same-position readout, not the recurrence.
    - ``hook_ssm_write`` — per-step write influence: a real HookPoint for the input-
      linear families (Mamba-1/2 ``dt·(x⊗B)``, added by the subclass), an alias onto
      ``hook_beta`` for the state-dependent delta rule; patching it re-runs the scan.

    ``eager_scan`` (opt-in, prefill only) swaps HF's fused kernel for a Python scan so
    these hooks fire; default False leaves the cached path bit-for-bit untouched. Must
    precede ``GeneralizedComponent`` in the bases so ``super().__init__`` reaches it.
    """

    eager_scan: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hook_ssm_state = HookPoint()


# Children present on *every* SSM2MixerBridge regardless of whether it wraps a
# real Mamba layer — universal I/O hooks, the wrapped module, and the eager-scan
# analysis hooks. None of these signal that the mixer is realized.
_PASSTHROUGH_CHILDREN = frozenset(
    {"hook_in", "hook_out", "_original_component", "hook_ssm_write", "hook_ssm_state"}
)


def _is_realized_ssm_mixer(mixer: object) -> bool:
    """True unless ``mixer`` is a no-op passthrough wrapper.

    A hybrid like NemotronH wires a single ``SSM2MixerBridge`` ``.mixer`` slot on
    *every* layer; on attention / MLP / MoE layers its optional projection
    submodules are skipped, leaving only the universal hooks. A realized mixer
    always has something more — projection submodules (Mamba-1/2) or interior
    hooks like ``hook_q`` / ``hook_log_decay`` (gated-delta-net). This structural
    check needs no ``cfg.layers_block_type``.
    """
    return any(name not in _PASSTHROUGH_CHILDREN for name in getattr(mixer, "_modules", {}))


def find_ssm_mixer(block: nn.Module) -> Optional[SSMMixerProtocol]:
    """Return the block's SSM mixer submodule, or None if it has none.

    Scans the layer-type variant slots (``.mixer`` / ``.linear_attn`` / ``.mamba``
    / ``.ssm``) and returns the first that conforms to ``SSMMixerProtocol`` and is
    a realized SSM mixer (not a passthrough wrapper). An attention layer (only
    ``.attn``), or a hybrid's passthrough ``.mixer`` on a non-SSM layer, returns
    None.
    """
    modules = getattr(block, "_modules", {})
    for name in VARIANT_SUBMODULE_NAMES:
        sub = modules.get(name)
        if isinstance(sub, SSMMixerProtocol) and _is_realized_ssm_mixer(sub):
            return sub
    return None
