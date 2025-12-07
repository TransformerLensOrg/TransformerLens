"""BLOOM-specific block bridge component.

BLOOM blocks require special arguments (alibi, attention_mask, residual) that standard
BlockBridge doesn't handle. This custom component generates and passes these arguments.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.block import BlockBridge


class BloomBlockBridge(BlockBridge):
    """Block bridge for BLOOM models that handles ALiBi positional encoding.

    BLOOM uses ALiBi (Attention with Linear Biases) instead of standard positional
    embeddings. This requires generating an alibi tensor and passing it to each block
    along with the attention_mask.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        """Initialize the BLOOM block bridge.

        Args:
            name: The name of the component in the model
            config: Model configuration (used to get n_heads for ALiBi)
            submodules: Dictionary of submodules to register
            hook_alias_overrides: Optional dictionary to override default hook aliases
        """
        super().__init__(name, config, submodules, hook_alias_overrides)
        self.config = config
        self._alibi_cache: Optional[torch.Tensor] = None

    @staticmethod
    def build_alibi_tensor(
        attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build ALiBi tensor for attention biasing.

        This is adapted from the HuggingFace BLOOM implementation.

        Args:
            attention_mask: Attention mask of shape [batch_size, seq_length]
            num_heads: Number of attention heads
            dtype: Data type for the tensor

        Returns:
            ALiBi tensor of shape [num_heads, 1, seq_length]
        """
        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2 ** torch.floor(
            torch.log2(torch.tensor(num_heads, dtype=torch.float32))
        )
        base = torch.tensor(
            2 ** (-(2 ** -(torch.log2(closest_power_of_2) - 3))), dtype=torch.float32
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2
                ** (
                    -(
                        2
                        ** -(
                            torch.log2(torch.tensor(2 * closest_power_of_2, dtype=torch.float32))
                            - 3
                        )
                    )
                ),
                dtype=torch.float32,
            )
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        # Shape: [num_heads, 1, seq_length]
        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None, None] * arange_tensor
        return alibi.to(dtype)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the BLOOM block.

        BLOOM blocks require `alibi` and `attention_mask` arguments. If the HF model's
        BloomModel.forward() is being called, it will generate these and pass them through.
        If they're missing (e.g., when called standalone), we generate them here.

        Args:
            *args: Positional arguments (first should be hidden_states)
            **kwargs: Keyword arguments

        Returns:
            Output from the original BLOOM block
        """
        # Debug: Check if alibi is being passed
        # print(f"BloomBlockBridge.forward() called with kwargs keys: {list(kwargs.keys())}")

        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Check if we should stop before executing this block
        if hasattr(self, "_stop_at_layer_idx") and self._stop_at_layer_idx is not None:
            import re

            if self.name is not None:
                match = (
                    re.search(r"blocks\.(\d+)", self.name)
                    or re.search(r"\.h\.(\d+)", self.name)
                    or re.search(r"\.layers\.(\d+)", self.name)
                )
            else:
                match = None
            if match:
                layer_idx = int(match.group(1))
                if layer_idx == self._stop_at_layer_idx:
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        input_tensor = args[0]
                    elif "hidden_states" in kwargs and isinstance(
                        kwargs["hidden_states"], torch.Tensor
                    ):
                        input_tensor = kwargs["hidden_states"]
                    else:
                        raise ValueError(f"Cannot find input tensor to stop at layer {layer_idx}")
                    input_tensor = self.hook_in(input_tensor)
                    from transformer_lens.model_bridge.exceptions import (
                        StopAtLayerException,
                    )

                    raise StopAtLayerException(input_tensor)

        # Apply hook_in to hidden_states
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        # BLOOM blocks require 'alibi' and 'attention_mask' arguments.
        # If HF's BloomModel.forward() is calling us, these will already be present.
        # Only generate them if they're missing (e.g., standalone block testing).
        if "alibi" not in kwargs or kwargs["alibi"] is None:
            # Get hidden_states to determine shape
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                hidden_states = args[0]
            elif "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                raise ValueError("Could not find hidden_states in args or kwargs")

            batch_size, seq_length, _ = hidden_states.shape
            device = hidden_states.device
            dtype = hidden_states.dtype

            # Generate attention_mask if missing
            if "attention_mask" not in kwargs or kwargs["attention_mask"] is None:
                # Create default attention mask (all ones)
                attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
            else:
                attention_mask = kwargs["attention_mask"]
                # Ensure it's 2D [batch, seq_length] for ALiBi generation
                if attention_mask.dim() == 4:
                    # If 4D, we need 2D version for ALiBi generation
                    # Extract the last row which tells us which positions are valid
                    attention_mask_2d = attention_mask[:, 0, -1, :].long()
                elif attention_mask.dim() == 2:
                    attention_mask_2d = attention_mask
                else:
                    raise ValueError(
                        f"Unexpected attention_mask dimensions: {attention_mask.dim()}"
                    )

            # Generate ALiBi bias
            if self.config and hasattr(self.config, "n_heads"):
                num_heads = self.config.n_heads
            else:
                # Fallback: try to infer from model
                num_heads = 16  # BLOOM-560M has 16 heads

            # Generate alibi
            alibi = self.build_alibi_tensor(attention_mask_2d, num_heads, dtype)

            # Add alibi to kwargs
            kwargs["alibi"] = alibi
        # else: alibi is already present from HF, don't overwrite it!

        # Call original component
        output = self.original_component(*args, **kwargs)

        # Apply hook_out
        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                first = self.hook_out(first)
                if len(output) == 1:
                    return first
                output = (first,) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            output = self.hook_out(output)
        return output
