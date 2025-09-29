"""Final normalization bridge component implementation."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


class FinalNormalizationBridge(NormalizationBridge):
    """Final layer normalization bridge that behaves as identity when weights are folded.

    This component extends NormalizationBridge and overrides the forward method to return
    identity (input unchanged) when layer norm folding is enabled, otherwise falls back
    to the standard normalization functionality.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the final normalization bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output or identity if folded
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # keep mypy happy
        assert self.config is not None

        # Check if layer norm folding is enabled - if so, behave as identity
        if hasattr(self.config, "layer_norm_folding") and self.config.layer_norm_folding:
            # Final layer norm becomes identity when folding is enabled
            # (weights are absorbed into other components during processing)
            # Simply return the input unchanged (identity function)
            return hidden_states
        else:
            # Fall back to standard normalization behavior
            return super().forward(hidden_states, **kwargs)

    @classmethod
    def create_final_normalization_bridge(
        cls,
        name: str,
        config: Any,
        original_component: Any,
    ) -> "FinalNormalizationBridge":
        """Create a final normalization bridge for final layer norm components.

        Args:
            name: The name of this component
            config: Configuration object
            original_component: The original layer norm component

        Returns:
            FinalNormalizationBridge that behaves as identity when folding is enabled
        """
        # Create the bridge
        bridge = cls(name=name, config=config)

        # Set the original component
        bridge.set_original_component(original_component)

        return bridge