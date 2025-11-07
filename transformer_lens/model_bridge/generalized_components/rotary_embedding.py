"""Rotary embedding bridge component.

This module contains the bridge component for rotary position embedding layers.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class RotaryEmbeddingBridge(GeneralizedComponent):
    """Rotary embedding bridge that wraps rotary position embedding layers.

    Unlike regular embeddings, rotary embeddings return a tuple of (cos, sin) tensors.
    This component properly handles the tuple return value without unwrapping it.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the rotary embedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for RotaryEmbeddingBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

        # Add separate hooks for cos and sin components
        self.hook_cos = HookPoint()
        self.hook_sin = HookPoint()

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the rotary embedding bridge.

        Rotary embeddings typically take seq_len or position_ids and return (cos, sin) tensors.

        Args:
            *args: Positional arguments to pass to the original component
            **kwargs: Keyword arguments to pass to the original component

        Returns:
            Tuple of (cos, sin) tensors for rotary position embeddings
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook to the first argument if it's a tensor
        if args and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]

        # Call the original component
        output = self.original_component(*args, **kwargs)

        # Rotary embeddings should return a tuple (cos, sin)
        # We don't unwrap it like regular embeddings do
        if not isinstance(output, tuple):
            # Some implementations might return just the tuple directly
            # Handle both old and new transformer versions
            if hasattr(output, "__iter__") and not isinstance(output, torch.Tensor):
                output = tuple(output)
            else:
                # Single tensor output - shouldn't happen but handle gracefully
                raise RuntimeError(
                    f"Rotary embedding {self.name} returned {type(output)} instead of tuple. "
                    f"Expected (cos, sin) tuple."
                )

        # Apply hooks to cos and sin separately
        # The tuple contains (cos, sin) tensors
        if len(output) == 2:
            cos, sin = output
            cos = self.hook_cos(cos)
            sin = self.hook_sin(sin)
            return (cos, sin)
        else:
            # Unexpected tuple length - just return as-is
            return output

    def get_dummy_inputs(
        self, test_input: torch.Tensor, **kwargs: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Generate dummy inputs for rotary embedding forward method.

        Rotary embeddings typically expect (x, position_ids) where:
        - x: input tensor [batch, seq, d_model]
        - position_ids: position indices [batch, seq]

        Args:
            test_input: Base test input tensor [batch, seq, d_model]
            **kwargs: Additional context including position_ids

        Returns:
            Tuple of (args, kwargs) for the rotary embedding forward method
        """
        batch, seq_len, _ = test_input.shape

        # Get position_ids from kwargs, or generate default
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=test_input.device).unsqueeze(0).expand(batch, -1)
            )

        # Rotary embeddings expect (x, position_ids)
        return (test_input, position_ids), {}
