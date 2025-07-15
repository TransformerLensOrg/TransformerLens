"""Layer norm bridge component implementation."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LayerNormBridge(GeneralizedComponent):
    """Layer norm bridge that wraps transformer layer normalization layers.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the layer norm bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for LayerNormBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)
        # No extra hooks; use only hook_in and hook_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the layer norm bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, **kwargs)
        output = self.hook_out(output)

        return output
