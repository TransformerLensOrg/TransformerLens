"""Gated RMS normalization bridge component.

Wraps Mamba-2's `MambaRMSNormGated`, which takes TWO positional inputs:
`(hidden_states, gate)`. Standard norm bridges assume a single-input signature,
so this bridge is needed to plumb both arguments through the wrapped module.

Hook layout:
    hook_in    — captures `hidden_states` (the value being normalized)
    hook_gate  — captures `gate` (the gating tensor)
    hook_out   — captures the gated normalized output
"""
from typing import Any, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class GatedRMSNormBridge(GeneralizedComponent):
    """Bridge component for Mamba-2's MambaRMSNormGated (two-input norm)."""

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
    ):
        super().__init__(name=name, config=config)
        # hook_in and hook_out are created by the parent. Add a dedicated
        # hook_gate for the gate tensor.
        self.hook_gate = HookPoint()

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass: hook hidden_states, gate, and the gated output.

        The underlying MambaRMSNormGated computes `rmsnorm(hidden_states * silu(gate))`.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        if gate is not None:
            gate = self.hook_gate(gate)

        output = self.original_component(hidden_states, gate, *args, **kwargs)
        return self.hook_out(output)
