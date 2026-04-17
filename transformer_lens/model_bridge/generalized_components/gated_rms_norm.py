"""Bridge for Mamba-2's MambaRMSNormGated — a norm that takes (hidden_states, gate)."""
from typing import Any, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class GatedRMSNormBridge(GeneralizedComponent):
    """Two-input norm wrapper. Exposes hook_in, hook_gate, hook_out.

    Standard norm bridges assume a single-input signature; this one threads
    both ``hidden_states`` and ``gate`` through the wrapped module.
    """

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
    ):
        super().__init__(name=name, config=config)
        self.hook_gate = HookPoint()

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
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
