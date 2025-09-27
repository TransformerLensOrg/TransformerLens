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

    @property
    def weight(self) -> torch.Tensor:
        """Return the embedding weight matrix (alias for W_E)."""
        return self.W_E

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

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process embedding weights according to GPT2 pretrained logic.

        For embeddings, this is a direct mapping without transformation.
        """
        if self.original_component is None:
            return

        # Determine the weight key based on the component name
        if "wte" in self.name or "embed" in self.name:
            weight_key = "W_E"
        elif "wpe" in self.name or "pos" in self.name:
            weight_key = "W_pos"
        else:
            # Default key
            weight_key = "W_E"

        # Store processed weights in TransformerLens format (direct mapping)
        self._processed_weights = {
            weight_key: self.original_component.weight,
        }

    def get_processed_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the processed weights in TransformerLens format.

        Returns:
            Dictionary mapping TransformerLens parameter names to processed tensors
        """
        if not hasattr(self, '_processed_weights') or self._processed_weights is None:
            # If weights haven't been processed, process them now
            self.process_weights()

        return self._processed_weights.copy()

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this embedding component.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        # Determine the weight key based on the component name (same logic as process_weights)
        if "wte" in self.name or "embed" in self.name:
            weight_key = "W_E"
        elif "wpe" in self.name or "pos" in self.name:
            weight_key = "W_pos"
        else:
            # Default key
            weight_key = "W_E"

        full_name = f"{prefix}.{weight_key}" if prefix else weight_key
        return [full_name]

    def custom_weight_processing(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        component_prefix: str,
        **processing_kwargs
    ) -> Dict[str, torch.Tensor]:
        """Custom weight processing for embeddings - direct mapping.

        Args:
            hf_state_dict: Raw HuggingFace state dict
            component_prefix: Prefix for this component's weights (e.g., "transformer.wte")
            **processing_kwargs: Additional processing arguments

        Returns:
            Dictionary of processed weights
        """
        processed_weights = {}

        # Determine weight key based on component name
        if "wte" in component_prefix or "embed" in self.name:
            weight_key = "W_E"
        elif "wpe" in component_prefix or "pos" in self.name:
            weight_key = "W_pos"
        else:
            weight_key = "W_E"

        # Direct mapping of embedding weights
        hf_weight_key = f"{component_prefix}.weight"
        if hf_weight_key in hf_state_dict:
            processed_weights[weight_key] = hf_state_dict[hf_weight_key]

        return processed_weights
