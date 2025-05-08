"""Bridge between HookedTransformer and underlying model architectures.

This module provides a pure adapter layer that maps HookedTransformer's component access patterns
to the underlying model's structure using architecture adapters.
"""

from typing import Any

import torch
from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_conversion import (
    ArchitectureConversion,
)


class TransformerBridge:
    """Bridge between HookedTransformer and underlying model architectures."""

    def __init__(
        self,
        model: PreTrainedModel,
        architecture_adapter: ArchitectureConversion,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the transformer bridge.

        Args:
            model: The underlying model to bridge to.
            architecture_adapter: The architecture adapter to use for mapping between models.
            device: The device to put the model on.
            dtype: The dtype to use for the model.
        """
        self.model = model
        self.architecture_adapter = architecture_adapter
        self.device = device
        self.dtype = dtype

    def __getattr__(self, name: str) -> Any:
        """Get a component from the model using the architecture adapter.

        This method allows HookedTransformer to access components using its own naming scheme,
        which are then mapped to the underlying model's structure using the architecture adapter.

        Args:
            name: The name of the component to get.

        Returns:
            The requested component.
        """
        # Use the architecture adapter to map the component name to the underlying model's structure
        return self.architecture_adapter.get_component(self.model, name)

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> "TransformerBridge":
        """Move the model to the specified device and dtype.

        Args:
            device: The device to move the model to.
            dtype: The dtype to convert the model to.

        Returns:
            self: The bridge instance.
        """
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        if dtype is not None:
            self.dtype = dtype
            self.model = self.model.to(dtype)
        return self 