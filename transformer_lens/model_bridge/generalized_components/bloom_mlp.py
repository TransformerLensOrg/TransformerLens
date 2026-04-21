"""BLOOM-specific MLP bridge component.

BLOOM MLP requires a special 'residual' argument that standard MLPBridge doesn't handle.
This custom component passes the residual argument through to the original component.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


class BloomMLPBridge(MLPBridge):
    """MLP bridge for BLOOM models that handles residual connections.

    BLOOM MLP has a unique forward signature that requires:
    - hidden_states (first positional arg)
    - residual (keyword arg): The residual connection tensor

    This bridge ensures the residual argument is properly passed through.
    """

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the BLOOM MLP bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration
            submodules: Dictionary of submodules to register (e.g., dense_h_to_4h, dense_4h_to_h)
        """
        super().__init__(name, config, submodules or {})

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through BLOOM MLP with hooks.

        BLOOM MLP requires these arguments:
        - hidden_states (first positional arg)
        - residual (second positional arg)

        Args:
            *args: Input arguments (hidden_states, residual)
            **kwargs: Additional keyword arguments (if any)

        Returns:
            Output tensor from BLOOM MLP
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply hook_in to hidden_states (first positional argument)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        # BLOOM MLP requires residual as second positional arg
        # The original BLOOM block passes it, so we just pass everything through
        # No need to validate since the original component will handle it

        # Call the original BLOOM MLP component with all arguments
        output = self.original_component(*args, **kwargs)

        # Apply hook_out
        output = self.hook_out(output)

        return output
