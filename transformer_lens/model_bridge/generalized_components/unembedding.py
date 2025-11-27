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

    property_aliases = {"W_U": "u.weight"}

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

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component and ensure it has bias enabled.

        Args:
            original_component: The original transformer component to wrap
        """
        # If this is a Linear layer without bias, enable it
        if isinstance(original_component, torch.nn.Linear) and original_component.bias is None:
            # Get the output features (vocab size)
            vocab_size = original_component.weight.shape[0]
            device = original_component.weight.device
            dtype = original_component.weight.dtype

            # Create a zero bias parameter
            original_component.bias = torch.nn.Parameter(
                torch.zeros(vocab_size, device=device, dtype=dtype)
            )

        super().set_original_component(original_component)

    @property
    def W_U(self) -> torch.Tensor:
        """Return the unembedding weight matrix in TL format [d_model, d_vocab]."""
        if "_processed_W_U" in self._parameters:
            processed_W_U = self._parameters["_processed_W_U"]
            if processed_W_U is not None:
                # Processed weights are in HF format [vocab, d_model]
                # Transpose to TL format [d_model, d_vocab]
                return processed_W_U.T
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has no weight attribute"
        weight = self.original_component.weight
        assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
        # HF format is [d_vocab, d_model], transpose to TL format [d_model, d_vocab]
        return weight.T

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the unembedding bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Unembedded output (logits)
        """
        # Otherwise delegate to original component
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            pass
        hidden_states = self.hook_in(hidden_states)
        if (
            target_dtype is not None
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.is_floating_point()
        ):
            hidden_states = hidden_states.to(dtype=target_dtype)
        output = self.original_component(hidden_states, **kwargs)
        output = self.hook_out(output)
        return output

    @property
    def b_U(self) -> torch.Tensor:
        """Access the unembedding bias vector."""
        if "_b_U" in self._parameters:
            param = self._parameters["_b_U"]
            if param is not None:
                return param
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        if hasattr(self.original_component, "bias") and self.original_component.bias is not None:
            bias = self.original_component.bias
            assert isinstance(bias, torch.Tensor), f"Bias is not a tensor for {self.name}"
            return bias
        else:
            assert hasattr(
                self.original_component, "weight"
            ), f"Component {self.name} has no weight attribute"
            weight = self.original_component.weight
            assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
            device = weight.device
            dtype = weight.dtype
            vocab_size: int = int(weight.shape[0])
            return torch.zeros(vocab_size, device=device, dtype=dtype)
