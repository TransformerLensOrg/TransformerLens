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
        self.hook_cos = HookPoint()
        self.hook_sin = HookPoint()

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Generate random inputs for rotary embedding testing.

        Rotary embeddings for Gemma-3 expect (x, position_ids) where:
        - x: tensor with shape [batch, seq, num_heads, head_dim]
        - position_ids: position indices with shape [batch, seq]

        Args:
            batch_size: Batch size for generated inputs
            seq_len: Sequence length for generated inputs
            device: Device to place tensors on
            dtype: Dtype for generated tensors

        Returns:
            Dictionary with positional args as tuple under 'args' key
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        if self.config and hasattr(self.config, "num_attention_heads"):
            num_heads = self.config.num_attention_heads
        else:
            num_heads = 4
        if self.config and hasattr(self.config, "head_dim"):
            head_dim = self.config.head_dim
        else:
            head_dim = 256
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        return {"args": (x, position_ids)}

    def forward(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the rotary embedding bridge.

        Rotary embeddings typically take seq_len or position_ids and return (cos, sin) tensors.
        This method ensures that cos and sin are passed through their respective hooks
        (hook_cos and hook_sin) to match HookedTransformer's behavior.

        Args:
            *args: Positional arguments to pass to the original component
            **kwargs: Keyword arguments to pass to the original component

        Returns:
            Tuple of (cos, sin) tensors for rotary position embeddings, after being
            passed through hook_cos and hook_sin respectively
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook if first arg is a tensor
        if args and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]

        # Call original component to get (cos, sin) tuple
        output = self.original_component(*args, **kwargs)

        # Ensure output is a tuple
        if not isinstance(output, tuple):
            if hasattr(output, "__iter__") and (not isinstance(output, torch.Tensor)):
                output = tuple(output)
            else:
                raise RuntimeError(
                    f"Rotary embedding {self.name} returned {type(output)} instead of tuple. Expected (cos, sin) tuple."
                )

        # Extract cos and sin, apply their respective hooks, and return
        if len(output) == 2:
            cos, sin = output
            # Apply hooks to match HookedTransformer's rotary_cos/rotary_sin pattern
            cos = self.hook_cos(cos)
            sin = self.hook_sin(sin)
            # Return the hooked cos and sin as a tuple
            # Note: Don't pass tuple through hook_out as it expects a tensor
            return (cos, sin)
        else:
            # For unexpected tuple lengths, just pass through
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
