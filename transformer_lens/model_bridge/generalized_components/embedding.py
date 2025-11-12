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
        # If using processed weights from compatibility mode, return those
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            if hasattr(self, "_processed_weight"):
                return self._processed_weight

        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")

        # Handle rotary embeddings (have inv_freq instead of weight)
        if hasattr(self.original_component, "inv_freq") and not hasattr(
            self.original_component, "weight"
        ):
            inv_freq = self.original_component.inv_freq
            assert isinstance(inv_freq, torch.Tensor), f"inv_freq is not a tensor for {self.name}"
            return inv_freq

        # Handle regular embeddings (have weight)
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has neither weight nor inv_freq attribute"
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

        # Check if we're using processed weights from a reference model (layer norm folding case)
        # This happens when _port_embedding_components has been called
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            # Apply input hook
            input_ids = self.hook_in(input_ids)

            # Use the processed weight directly with F.embedding
            if hasattr(self, "_processed_weight"):
                output = torch.nn.functional.embedding(input_ids, self._processed_weight)
            else:
                # Fallback to original component's weight
                output = torch.nn.functional.embedding(input_ids, self.W_E)

            # Apply output hook
            output = self.hook_out(output)

            return output

        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Get the target dtype from the original component's weight
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            # Component has no parameters, keep inputs as-is
            pass

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

        # Some models return tuples; extract embeddings
        if isinstance(output, tuple):
            output = output[0]

        # Ensure output dtype matches original component's dtype
        if target_dtype is not None and output.dtype != target_dtype:
            output = output.to(dtype=target_dtype)

        # Apply output hook
        output = self.hook_out(output)

        return output

    def set_processed_weight(self, weight: torch.Tensor) -> None:
        """Set the processed weight to use when layer norm is folded.

        Args:
            weight: The processed embedding weight tensor
        """
        self._processed_weight = weight
        self._use_processed_weights = True
