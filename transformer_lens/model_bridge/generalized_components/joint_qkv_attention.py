"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused QKV matrix.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class JointQKVAttentionBridge(AttentionBridge):
    """Joint QKV attention bridge that wraps a joint QKV linear layer.

    This component wraps attention layers that use a fused QKV matrix such that both
    the activations from the joint QKV matrix and from the individual, separated Q, K, and V matrices
    are hooked and accessible.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the joint QKV attention bridge.

        Args:
            name: The name of this component
            config: Configuration (split_qkv_matrix function is required)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, submodules=submodules)

        self.config = config
        if self.config is None:
            raise RuntimeError(
                f"Config not set for {self.name}. Config is required for QKV separation."
            )
        if "split_qkv_matrix" not in self.config:
            raise RuntimeError(f"Config for {self.name} must include 'split_qkv_matrix' function.")

        # Create LinearBridge components for Q, K, and V activations
        self.q = LinearBridge(name="q")
        self.k = LinearBridge(name="k")
        self.v = LinearBridge(name="v")

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps and initialize LinearBridges for Q, K, and V transformations.

        Args:
            original_component: The original attention layer to wrap
        """

        super().set_original_component(original_component)

        # Keep mypy happy
        assert self.config is not None

        W_Q_transformation, W_K_transformation, W_V_transformation = self.config[
            "split_qkv_matrix"
        ](original_component)

        # Initialize LinearBridges for Q, K, and V transformations
        self.q.set_original_component(W_Q_transformation)
        self.k.set_original_component(W_K_transformation)
        self.v.set_original_component(W_V_transformation)

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the QKV linear transformation with hooks.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after QKV linear transformation
        """
        output = super().forward(input, *args, **kwargs)

        # Run the input through the individual Q, K, and V transformations
        # in order to hook their outputs
        self.q(input)
        self.k(input)
        self.v(input)

        return output
