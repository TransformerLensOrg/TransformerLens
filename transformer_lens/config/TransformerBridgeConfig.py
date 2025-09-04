"""Configuration class for TransformerBridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .TransformerLensConfig import TransformerLensConfig


@dataclass
class TransformerBridgeConfig(TransformerLensConfig):
    """
    Configuration for TransformerBridge.

    This extends TransformerLensConfig with bridge-specific properties,
    particularly architecture information needed for adapter selection.
    """

    # Architecture information for adapter selection
    architecture: Optional[str] = None

    # Tokenizer configuration
    tokenizer_prepends_bos: bool = True
    default_padding_side: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()

        # Validate architecture if provided
        if self.architecture is not None and not isinstance(self.architecture, str):
            raise ValueError(f"architecture must be a string, got {type(self.architecture)}")
