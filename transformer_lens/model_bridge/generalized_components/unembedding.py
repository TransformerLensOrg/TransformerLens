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

    property_aliases = {
        "W_U": "u.weight",
    }

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

    @property
    def W_U(self) -> torch.Tensor:
        """Return the unembedding weight matrix."""
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has no weight attribute"
        weight = self.original_component.weight
        assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
        return weight.T

    @property
    def weight(self) -> torch.Tensor:
        """Return the unembedding weight matrix (alias for W_U)."""
        return self.W_U

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

    @property
    def b_U(self) -> torch.Tensor:
        """Access the unembedding bias vector."""
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")

        # Handle case where the original component doesn't have a bias (like GPT-2)
        if hasattr(self.original_component, "bias") and self.original_component.bias is not None:
            bias = self.original_component.bias
            assert isinstance(bias, torch.Tensor), f"Bias is not a tensor for {self.name}"
            return bias
        else:
            # Return zero bias of appropriate shape [d_vocab]
            assert hasattr(
                self.original_component, "weight"
            ), f"Component {self.name} has no weight attribute"
            weight = self.original_component.weight
            assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
            device = weight.device
            dtype = weight.dtype
            vocab_size: int = int(weight.shape[0])  # lm_head weight is [d_vocab, d_model]
            return torch.zeros(vocab_size, device=device, dtype=dtype)

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process unembedding weights according to GPT2 pretrained logic.

        The unembedding weight processing involves transposing the lm_head weight,
        which is already handled in the W_U property.
        """
        # Store processed weights in TransformerLens format
        if self.original_component is None:
            return

        self._processed_weights = {
            "W_U": self.W_U,  # This already applies the transpose
            "b_U": self.b_U,  # This handles bias or zero bias appropriately
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
        """Get the expected TransformerLens parameter names for this unembedding component.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        # Unembedding components always have W_U and b_U (bias is zero if not present)
        w_name = f"{prefix}.W_U" if prefix else "W_U"
        b_name = f"{prefix}.b_U" if prefix else "b_U"
        return [w_name, b_name]
