"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused QKV matrix.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class QKVHooks(torch.nn.Module):
    """Container for Q, K, or V hook points."""

    def __init__(self):
        super().__init__()
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()


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

        # Create hook points for individual Q, K and V activations
        self.W_Q = QKVHooks()
        self.W_K = QKVHooks()
        self.W_V = QKVHooks()

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

        # Keep mypy happy
        assert self.config is not None

        W_Q_transformation, W_K_transformation, W_V_transformation = self.config[
            "split_qkv_matrix"
        ](self)

        # Apply Q hook
        output_Q = self.W_Q.hook_in(W_Q_transformation(input))
        output_Q = self.W_Q.hook_out(output_Q)

        # Apply K hook
        output_K = self.W_K.hook_in(W_K_transformation(input))
        output_K = self.W_K.hook_out(output_K)

        # Apply V hook
        output_V = self.W_V.hook_in(W_V_transformation(input))
        output_V = self.W_V.hook_out(output_V)

        return output
