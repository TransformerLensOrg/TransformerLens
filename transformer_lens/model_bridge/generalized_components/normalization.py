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
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the normalization bridge.

        Args:
            name: The name of this component
            config: Optional configuration
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
<<<<<<< HEAD
        x = self.hook_in(x)
        x = x.to(torch.float32)
=======
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)

        if not self.config.uses_rms_norm:
            # Only center if not using RMSNorm
            hidden_states = hidden_states - hidden_states.mean(-1, keepdim=True)

        scale = self.hook_scale(
            (hidden_states.pow(2).mean(-1, keepdim=True) + self.config.eps).sqrt()
        )
        hidden_states = self.hook_normalized(hidden_states / scale)

        if self.config.uses_rms_norm:
            # No bias if using RMSNorm
            output = hidden_states * self.weight
        else:
            # Add bias if using LayerNorm
            output = hidden_states * self.weight + self.bias
>>>>>>> 668af11b (Add configuration dictionary during initialization)

        x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
        scale: Union[
            Float[torch.Tensor, "batch pos 1"],
            Float[torch.Tensor, "batch pos head_index 1"],
        ] = (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
        output =  (x / scale)
        output = self.hook_out(output)
        return output
