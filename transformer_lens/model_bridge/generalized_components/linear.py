"""Linear bridge component for wrapping linear layers with hook points."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LinearBridge(GeneralizedComponent):
    """Bridge component for linear layers.

    This component wraps a linear layer (nn.Linear) and provides hook points
    for intercepting the input and output activations.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ) -> None:
        """Initialize the LinearBridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for LinearBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the linear layer with hooks.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after linear transformation
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook
        input = self.hook_in(input)

        # Forward through the original linear layer
        output = self.original_component(input, *args, **kwargs)

        # Apply output hook
        output = self.hook_out(output)

        return output

    @property
    def in_features(self) -> int:
        """Get input features from the original component."""
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert isinstance(self.original_component, torch.nn.Linear)
        return self.original_component.in_features

    @property
    def out_features(self) -> int:
        """Get output features from the original component."""
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert isinstance(self.original_component, torch.nn.Linear)
        return self.original_component.out_features

    @property
    def bias(self) -> bool:
        """Check if the original component has bias."""
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        return self.original_component.bias is not None

    def __repr__(self) -> str:
        """String representation of the LinearBridge."""
        if self.original_component is not None:
            return f"LinearBridge({self.in_features} -> {self.out_features}, bias={self.bias})"
        else:
            return f"LinearBridge(name={self.name}, original_component=None)"
