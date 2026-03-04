"""Vision Projection bridge component.

This module contains the bridge component for multimodal projection layers
that map vision encoder outputs to the language model's embedding space.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class VisionProjectionBridge(GeneralizedComponent):
    """Bridge for the multimodal projection layer.

    This component bridges vision encoder outputs to language model inputs.
    In Gemma 3, this is the `multi_modal_projector` which contains:
    - mm_soft_emb_norm: RMSNorm for normalizing vision embeddings
    - avg_pool: Average pooling to reduce spatial dimensions

    The projection maps vision_hidden_size -> language_hidden_size.
    """

    hook_aliases = {
        "hook_vision_proj_in": "hook_in",
        "hook_vision_proj_out": "hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the vision projection bridge.

        Args:
            name: The name of this component (e.g., "multi_modal_projector")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

        # Hook for after projection before it's combined with text
        self.hook_projected = HookPoint()

    def forward(
        self,
        vision_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the vision projection.

        Args:
            vision_features: Vision encoder output [batch, num_patches, vision_hidden_size]
            **kwargs: Additional arguments

        Returns:
            Projected features [batch, num_tokens, language_hidden_size]
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook
        vision_features = self.hook_in(vision_features)

        # Forward through the projection layer
        output = self.original_component(vision_features, **kwargs)

        # Apply output hook
        if isinstance(output, tuple):
            output = (self.hook_out(output[0]),) + output[1:]
        else:
            output = self.hook_out(output)

        return output
