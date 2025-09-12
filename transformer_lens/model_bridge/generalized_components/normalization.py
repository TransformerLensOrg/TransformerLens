"""Normalization bridge component implementation."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class NormalizationBridge(GeneralizedComponent):
    """Normalization bridge that wraps transformer normalization layers but implements the calculation from scratch.

    This component provides standardized input/output hooks.
    """

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

        self.hook_normalized = HookPoint()
        self.hook_scale = HookPoint()

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the normalization bridge.

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

        # keep mypy happy
        assert self.config is not None

        hidden_states = self.hook_in(hidden_states)

        if not self.config.uses_rms_norm:
            # Only center if not using RMSNorm
            hidden_states = hidden_states - hidden_states.mean(-1, keepdim=True)

        scale = self.hook_scale(
            (hidden_states.pow(2).mean(-1, keepdim=True) + self.config.eps).sqrt()
        )
        hidden_states = self.hook_normalized(hidden_states / scale)

        if not self.config.layer_norm_folding:
            if self.config.uses_rms_norm:
                # No bias if using RMSNorm
                hidden_states = hidden_states * self.weight
            else:
                # Add bias if using LayerNorm
                hidden_states = hidden_states * self.weight + self.bias

        output = self.hook_out(hidden_states)
        return output
