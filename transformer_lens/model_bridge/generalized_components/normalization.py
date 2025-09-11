"""Normalization bridge component implementation."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class NormalizationBridge(GeneralizedComponent):
    """Normalization bridge that wraps transformer normalization layers.

    This component provides standardized input/output hooks.
    """

    hook_aliases = {"hook_normalized": "hook_out", "hook_scale": "hook_out"}

    property_aliases = {
        "w": "weight",
        "b": "bias",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the normalization bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for NormalizationBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)
        # No extra hooks; use only hook_in and hook_out

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the normalization bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output
        """
        x = self.hook_in(x)
        x = x.to(torch.float32)

        x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
        scale: Union[
            Float[torch.Tensor, "batch pos 1"],
            Float[torch.Tensor, "batch pos head_index 1"],
        ] = (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
        output =  (x / scale)
        output = self.hook_out(output)
        return output
