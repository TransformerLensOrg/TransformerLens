"""Unembedding bridge component.

This module contains the bridge component for unembedding layers.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class UnembeddingBridge(GeneralizedComponent):
    """Unembedding bridge that wraps transformer unembedding layers.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the unembedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for UnembeddingBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)
        # No extra hooks; use only hook_in and hook_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the unembedding bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Unembedded output (logits)
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, **kwargs)
        output = self.hook_out(output)

        return output
