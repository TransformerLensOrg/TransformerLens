"""Positional embedding bridge component.

This module contains the bridge component for positional embedding layers.
"""

import inspect
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class PosEmbedBridge(GeneralizedComponent):
    """Positional embedding bridge that wraps transformer positional embedding layers.

    This component provides standardized input/output hooks for positional embeddings.
    """

    property_aliases = {
        "W_pos": "weight",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the positional embedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for PosEmbedBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)
        # No extra hooks; use only hook_in and hook_out

    @property
    def W_pos(self) -> torch.Tensor:
        """Return the positional embedding weight matrix."""
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
        """Forward pass through the positional embedding bridge.

        Args:
            input_ids: Input token IDs (used to determine sequence length and batch size)
            position_ids: Optional position IDs, if None will generate them automatically
            **kwargs: Additional arguments

        Returns:
            Positional embeddings
        """
        # Check if we're using processed weights from a reference model (layer norm folding case)
        if hasattr(self, '_use_processed_weights') and self._use_processed_weights:
            # Apply input hook to input_ids (for consistency, though pos embed doesn't really use input_ids)
            input_ids = self.hook_in(input_ids)

            # Generate position indices if not provided
            if position_ids is None:
                batch_size, seq_len = input_ids.shape[:2]
                position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            # Use the processed weight directly with F.embedding
            if hasattr(self, '_processed_weight'):
                output = torch.nn.functional.embedding(position_ids, self._processed_weight)
            else:
                # Fallback to original component's weight
                output = torch.nn.functional.embedding(position_ids, self.W_pos)

            # Apply output hook
            output = self.hook_out(output)

            return output

        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook to input_ids
        input_ids = self.hook_in(input_ids)

        # For standard positional embeddings, we need to generate position indices
        if position_ids is None:
            batch_size, seq_len = input_ids.shape[:2]
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Check if the original component supports position_ids using inspect.signature
        sig = inspect.signature(self.original_component.forward)
        supports_position_ids = "position_ids" in sig.parameters

        if not hasattr(self.original_component, "forward") or not supports_position_ids:
            # For simple embedding layers, call directly with position_ids
            output = self.original_component(position_ids, **kwargs)
        else:
            output = self.original_component(position_ids=position_ids, **kwargs)

        # Apply output hook
        output = self.hook_out(output)

        return output

    def set_processed_weight(self, weight: torch.Tensor) -> None:
        """Set the processed weight to use when layer norm is folded.

        Args:
            weight: The processed positional embedding weight tensor
        """
        self._processed_weight = weight
        self._use_processed_weights = True