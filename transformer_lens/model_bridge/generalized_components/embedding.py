"""Embedding bridge component.

This module contains the bridge component for embedding layers.
"""

import inspect
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class EmbeddingBridge(GeneralizedComponent):
    """Embedding bridge that wraps transformer embedding layers.

    This component provides standardized input/output hooks.
    """

    property_aliases = {
        "W_E": "e.weight",
        "W_pos": "pos.weight",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the embedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for EmbeddingBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)
        # No extra hooks; use only hook_in and hook_out

    @property
    def W_E(self) -> torch.Tensor:
        """Return the embedding weight matrix."""
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has no weight attribute"
        weight = self.original_component.weight
        assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
        return weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the embedding bridge.

        Args:
            input_ids: Input token IDs
            position_ids: Optional position IDs (ignored if not supported)
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Embedded output
        """

        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook
        input_ids = self.hook_in(input_ids)

        # Check if the original component supports position_ids using inspect.signature
        sig = inspect.signature(self.original_component.forward)
        supports_position_ids = "position_ids" in sig.parameters

        if not hasattr(self.original_component, "forward") or not supports_position_ids:
            kwargs.pop("position_ids", None)
            output = self.original_component(input_ids, **kwargs)
        else:
            output = self.original_component(input_ids, position_ids=position_ids, **kwargs)

        # Apply output hook
        output = self.hook_out(output)

        return output
