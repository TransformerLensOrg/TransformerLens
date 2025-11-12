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

        # Get model dimensions from config if available
        if self.config and hasattr(self.config, "num_attention_heads"):
            num_heads = self.config.num_attention_heads
        else:
            num_heads = 4  # fallback

        if self.config and hasattr(self.config, "head_dim"):
            head_dim = self.config.head_dim
        else:
            head_dim = 256  # fallback

        # Create dummy x tensor (like Q or K)
        x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        # Create position_ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Return as positional args (not kwargs, since Gemma3RotaryEmbedding uses positional)
        return {"args": (x, position_ids)}

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
