"""SigLIP Vision Encoder bridge component.

This module contains the bridge component for SigLIP vision encoder layers
used in multimodal models like Gemma 3 and MedGemma.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


class SiglipVisionEncoderLayerBridge(GeneralizedComponent):
    """Bridge for a single SigLIP encoder layer.

    SigLIP encoder layers have:
    - layer_norm1: LayerNorm
    - self_attn: SiglipAttention
    - layer_norm2: LayerNorm
    - mlp: SiglipMLP
    """

    is_list_item: bool = True
    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
        "hook_attn_in": "attn.hook_in",
        "hook_attn_out": "attn.hook_out",
        "hook_mlp_in": "mlp.hook_in",
        "hook_mlp_out": "mlp.hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the SigLIP encoder layer bridge.

        Args:
            name: The name of this component (e.g., "encoder.layers")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the vision encoder layer.

        Args:
            hidden_states: Input hidden states from previous layer
            **kwargs: Additional arguments (attention_mask, etc.)

        Returns:
            Output hidden states
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, **kwargs)

        if isinstance(output, tuple):
            output = (self.hook_out(output[0]),) + output[1:]
        else:
            output = self.hook_out(output)

        return output


class SiglipVisionEncoderBridge(GeneralizedComponent):
    """Bridge for the complete SigLIP vision encoder.

    The SigLIP vision tower consists of:
    - vision_model.embeddings: Patch + position embeddings
    - vision_model.encoder.layers[]: Stack of encoder layers
    - post_layernorm: Final layer norm

    This bridge wraps the entire vision tower to provide hooks for
    interpretability of the vision processing pipeline.
    """

    hook_aliases = {
        "hook_vision_embed": "embeddings.hook_out",
        "hook_vision_out": "hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the SigLIP vision encoder bridge.

        Args:
            name: The name of this component (e.g., "vision_tower")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        default_submodules = {
            "embeddings": GeneralizedComponent(name="vision_model.embeddings"),
            "encoder_layers": SiglipVisionEncoderLayerBridge(name="vision_model.encoder.layers"),
            "post_layernorm": NormalizationBridge(
                name="vision_model.post_layernorm", config=config
            ),
        }

        if submodules:
            default_submodules.update(submodules)

        super().__init__(name, config, submodules=default_submodules)

        # Additional hooks for vision-specific processing
        self.hook_patch_embed = HookPoint()  # After patch embedding
        self.hook_pos_embed = HookPoint()  # After position embedding added

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            pixel_values: Input image tensor [batch, channels, height, width]
            **kwargs: Additional arguments

        Returns:
            Vision embeddings [batch, num_patches, hidden_size]
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook to pixel values
        pixel_values = self.hook_in(pixel_values)

        # Forward through the vision tower
        output = self.original_component(pixel_values, **kwargs)

        # Handle tuple output (some models return (hidden_states, ...))
        if isinstance(output, tuple):
            output = (self.hook_out(output[0]),) + output[1:]
        elif hasattr(output, "last_hidden_state"):
            # Handle BaseModelOutput-like returns
            output.last_hidden_state = self.hook_out(output.last_hidden_state)
        else:
            output = self.hook_out(output)

        return output
