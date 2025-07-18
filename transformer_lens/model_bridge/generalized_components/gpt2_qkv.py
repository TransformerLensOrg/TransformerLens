"""GPT-2 QKV bridge component.

This module contains the bridge component for GPT-2 QKV layers.
"""

from typing import Any, Dict, Optional

import einops
import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class GPT2QKVBridge(GeneralizedComponent):
    """GPT-2 QKV bridge that wraps GPT2's joint QKV linear layer.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the GPT-2 QKV bridge.

        Args:
            name: The name of this component
            config: Configuration
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)
        # Standardized hooks for all bridge components
        self.W_Q_hook_in = HookPoint()
        self.W_Q_hook_out = HookPoint()
        self.W_K_hook_in = HookPoint()
        self.W_K_hook_out = HookPoint()
        self.W_V_hook_in = HookPoint()
        self.W_V_hook_out = HookPoint()

    def split_qkv_matrix(self) -> tuple[torch.nn.Linear, torch.nn.Linear, torch.nn.Linear]:
        """Split the QKV matrix into separate linear transformations.
        Args:
            weights: The weight matrix of the original QKV layer
        Returns:
            Tuple of nn.Linear modules for Q, K, and V transformations
        """

        if self.config is None:
            raise RuntimeError(
                f"Config not set for {self.name}. Config is required for QKV matrix splitting."
            )

        # Keep mypy happy
        assert self.original_component is not None

        weights = self.original_component.weight

        # Keep mypy happy
        assert isinstance(weights, torch.Tensor)

        W_Q, W_K, W_V = torch.tensor_split(weights, 3, dim=1)

        bias_tensor = einops.rearrange(
            self.original_component.bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.config.n_head,
            head=self.config.n_embd // self.config.n_head,
        )

        # Keep mypy happy
        assert isinstance(bias_tensor, torch.Tensor)

        b_q, b_k, b_v = bias_tensor

        # Create nn.Linear module
        W_Q_transformation = torch.nn.Linear(W_Q.shape[0], W_Q.shape[1], bias=False)

        # Set the weight and bias
        W_Q_transformation.weight = torch.nn.Parameter(W_Q.T)
        W_Q_transformation.bias = torch.nn.Parameter(b_q.flatten())

        # Create nn.Linear module for K
        W_K_transformation = torch.nn.Linear(W_K.shape[0], W_K.shape[1], bias=False)

        # Set the weight and bias
        W_K_transformation.weight = torch.nn.Parameter(W_K.T)
        W_K_transformation.bias = torch.nn.Parameter(b_k.flatten())

        # Create nn.Linear module for V
        W_V_transformation = torch.nn.Linear(W_V.shape[0], W_V.shape[1], bias=False)

        # Set the weight and bias
        W_V_transformation.weight = torch.nn.Parameter(W_V.T)
        W_V_transformation.bias = torch.nn.Parameter(b_v.flatten())

        return W_Q_transformation, W_K_transformation, W_V_transformation

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the QKV linear transformation with hooks.

        Args:
            input: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after QKV linear transformation
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook
        input = self.hook_in(input)

        # Forward through the original linear layer
        output = self.original_component(input, *args, **kwargs)

        # Split the QKV matrix into separate transformations for investigating each activation
        W_Q_transformation, W_K_transformation, W_V_transformation = self.split_qkv_matrix()

        # Apply Q hook
        output_Q = self.W_Q_hook_in(W_Q_transformation(input))
        output_Q = self.W_Q_hook_out(output_Q)

        # Apply K hook
        output_K = self.W_K_hook_in(W_K_transformation(input))
        output_K = self.W_K_hook_out(output_K)

        # Apply V hook
        output_V = self.W_V_hook_in(W_V_transformation(input))
        output_V = self.W_V_hook_out(output_V)

        # Apply output hook
        output = self.hook_out(output)

        return output
