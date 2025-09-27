"""Configuration class for TransformerBridge."""

from dataclasses import dataclass
from typing import Optional

from .TransformerLensConfig import TransformerLensConfig


class TransformerBridgeConfig(TransformerLensConfig):
    """
    Configuration for TransformerBridge.

    This extends TransformerLensConfig with bridge-specific properties,
    particularly architecture information needed for adapter selection.
    """

    def __init__(self,
                 d_model: int,
                 d_head: int,
                 n_layers: int,
                 n_ctx: int,
                 architecture: Optional[str] = None,
                 tokenizer_prepends_bos: bool = True,
                 default_padding_side: Optional[str] = None,
                 **kwargs):
        """Initialize TransformerBridgeConfig."""
        super().__init__(d_model=d_model, d_head=d_head, n_layers=n_layers, n_ctx=n_ctx, **kwargs)

        # Architecture information for adapter selection
        self.architecture = architecture

        # Tokenizer configuration
        self.tokenizer_prepends_bos = tokenizer_prepends_bos
        self.default_padding_side = default_padding_side

        # Attention weight processing configuration
        self.split_attention_weights = False

        self.__post_init__()

    def __post_init__(self):
        """Post-initialization processing."""
        # Validate architecture if provided before calling super()
        if hasattr(self, 'architecture') and self.architecture is not None and not isinstance(self.architecture, str):
            raise ValueError(f"architecture must be a string, got {type(self.architecture)}")

        # Call parent's __post_init__ after our validation
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
