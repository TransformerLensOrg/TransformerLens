"""Mixture of Experts bridge component.

This module contains the bridge component for Mixture of Experts layers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MoEBridge(GeneralizedComponent):
    """Bridge component for Mixture of Experts layers.

    This component wraps a Mixture of Experts layer from a remote model and provides a consistent interface
    for accessing its weights and performing MoE operations.

    MoE models often return tuples of (hidden_states, router_scores). This bridge handles that pattern
    and provides a hook for capturing router scores.
    """

    # Hook aliases for compatibility with HookedTransformer naming
    hook_aliases = {
        "hook_pre": "hook_in",  # Pre-MoE activation
        "hook_post": "hook_out",  # Post-MoE activation (same as mlp.hook_out)
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the MoE bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for MoEBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)

        # Add hook for router scores (expert selection probabilities)
        self.hook_router_scores = HookPoint()

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Generate random inputs for component testing.

        Args:
            batch_size: Batch size for generated inputs
            seq_len: Sequence length for generated inputs
            device: Device to place tensors on
            dtype: Dtype for generated tensors (defaults to float32)

        Returns:
            Dictionary of input tensors matching the component's expected input signature
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        # MoE layers typically just need hidden_states as input
        # Use config.d_model if available, otherwise use a default
        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 768

        return {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the MoE bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            Same return type as original component (tuple or tensor).
            For MoE models that return (hidden_states, router_scores), preserves the tuple.
            Router scores are also captured via hook for inspection.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Get the target dtype from the original component's parameters
        target_dtype = None
        try:
            target_dtype = next(self.original_component.parameters()).dtype
        except StopIteration:
            # Component has no parameters, keep inputs as-is
            pass

        # Apply input hook and dtype conversion
        if len(args) > 0:
            hooked = self.hook_in(args[0])
            # Cast to target dtype if needed and input is a float tensor
            if (
                target_dtype is not None
                and isinstance(hooked, torch.Tensor)
                and hooked.is_floating_point()
            ):
                hooked = hooked.to(dtype=target_dtype)
            args = (hooked,) + args[1:]

        # Call the original MoE component
        output = self.original_component(*args, **kwargs)

        # Handle MoE models that return (hidden_states, router_scores) tuples
        # Most MoE implementations return tuples for diagnostic purposes
        if isinstance(output, tuple):
            hidden_states = output[0]

            # If router scores are present, capture them via hook
            if len(output) > 1:
                router_scores = output[1]
                # Apply router scores hook to allow inspection of expert routing
                self.hook_router_scores(router_scores)

            # Apply output hook to hidden states
            hidden_states = self.hook_out(hidden_states)

            # Preserve original return signature (tuple) to maintain compatibility
            # with HuggingFace model code that expects tuple unpacking
            return (hidden_states,) + output[1:]
        else:
            # Non-tuple output (fallback for non-MoE or different MLP types)
            hidden_states = self.hook_out(output)
            return hidden_states
