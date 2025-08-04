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
