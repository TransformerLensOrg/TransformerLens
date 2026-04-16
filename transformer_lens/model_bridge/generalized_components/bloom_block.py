"""BLOOM-specific block bridge component.

BLOOM blocks require special arguments (alibi, attention_mask, residual) that standard
BlockBridge doesn't handle. This custom component generates and passes these arguments.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.alibi_utils import (
    build_alibi_tensor as _build_alibi_tensor,
)
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

    @staticmethod
    def build_alibi_tensor(
        attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build ALiBi tensor for attention biasing.

        Delegates to the shared ALiBi utility in alibi_utils.py.

        Args:
            attention_mask: Attention mask of shape [batch_size, seq_length]
            num_heads: Number of attention heads
            dtype: Data type for the tensor

        Returns:
            ALiBi tensor of shape [batch_size, num_heads, 1, seq_length].
        """
        return _build_alibi_tensor(attention_mask, num_heads, dtype)

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

        self._check_stop_at_layer(*args, **kwargs)
        args, kwargs = self._hook_input_hidden_states(args, kwargs)

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

            # Generate alibi — shared utility returns [batch, heads, 1, seq],
            # reshape to [batch*heads, 1, seq] to match HF's format for baddbmm.
            alibi = self.build_alibi_tensor(attention_mask_2d, num_heads, dtype)
            alibi = alibi.reshape(batch_size * num_heads, 1, seq_length)

            # Add alibi to kwargs
            kwargs["alibi"] = alibi
        # else: alibi is already present from HF, don't overwrite it!

        # Call original component
        output = self.original_component(*args, **kwargs)
        return self._apply_output_hook(output, wrap_single_element=False)
