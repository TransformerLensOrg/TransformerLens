"""Parameter processing conversion for state dict transformations."""

import re
from typing import Dict, Optional

import torch

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)


class ParamProcessingConversion:
    """Handles conversion of parameters in state dicts with optional source key mapping.

    This class wraps a TensorConversion and manages fetching tensors from the state dict,
    applying conversions, and storing results back.

    Args:
        tensor_conversion: The conversion to apply to the tensor
        source_key: Optional source key template for fetching the tensor.
                   If not provided, uses the current key passed to convert().
                   Supports placeholders like {i} for layer indices.
    """

    def __init__(
        self,
        tensor_conversion: BaseTensorConversion,
        source_key: Optional[str] = None,
    ):
        self.tensor_conversion = tensor_conversion
        self.source_key = source_key

    def _resolve_key(self, current_key: str, template_key: str) -> str:
        """Resolve template key by extracting indices from current key.

        Args:
            current_key: The current key (e.g., "blocks.5.attn.q.weight")
            template_key: Template with placeholders (e.g., "blocks.{i}.attn.qkv.weight")

        Returns:
            Resolved key with placeholders filled in
        """
        # Extract layer index from current key if present
        layer_match = re.search(r"blocks\.(\d+)\.", current_key)
        if layer_match and "{i}" in template_key:
            layer_idx = layer_match.group(1)
            return template_key.replace("{i}", layer_idx)
        return template_key

    def convert(self, state_dict: Dict[str, torch.Tensor], current_key: str) -> torch.Tensor:
        """Convert a parameter in the state dict.

        Fetches tensor from source_key (or current_key if not specified),
        applies conversion, and stores result at current_key.

        Args:
            state_dict: The state dictionary to modify
            current_key: The key where the converted tensor should be stored

        Returns:
            Modified state dictionary
        """
        # Determine which key to fetch from
        fetch_key = current_key
        if self.source_key is not None:
            fetch_key = self._resolve_key(current_key, self.source_key)

        # Fetch tensor (may be None for optional parameters)
        tensor = state_dict.get(fetch_key)

        # Apply conversion (handles None gracefully)
        return self.tensor_conversion.convert(tensor, state_dict)

    def revert(self, tensor: torch.Tensor) -> torch.Tensor:
        """Revert a parameter conversion in the state dict.

        Fetches tensor from current_key, applies reversion,
        and stores result back at current_key.

        Args:
            state_dict: The state dictionary to modify
            current_key: The key of the tensor to revert

        Returns:
            Modified state dictionary
        """
        # Apply reversion (handles None gracefully)
        return self.tensor_conversion.revert(tensor)
