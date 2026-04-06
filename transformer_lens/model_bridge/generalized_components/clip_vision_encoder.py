"""CLIP Vision Encoder bridge component.

This module contains the bridge component for CLIP vision encoder layers
used in multimodal models like LLava.
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


class CLIPVisionEncoderLayerBridge(GeneralizedComponent):
    """Bridge for a single CLIP encoder layer.

    CLIP encoder layers have:
    - layer_norm1: LayerNorm
    - self_attn: CLIPAttention
    - layer_norm2: LayerNorm
    - mlp: CLIPMLP
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
        """Initialize the CLIP encoder layer bridge.

        Args:
            name: The name of this component (e.g., "encoder.layers")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the vision encoder layer.

        Args:
            hidden_states: Input hidden states from previous layer
            attention_mask: Optional attention mask
            causal_attention_mask: Optional causal attention mask (used by CLIP encoder)
            **kwargs: Additional arguments

        Returns:
            Output hidden states
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(
            hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            **kwargs,
        )

        if isinstance(output, tuple):
            output = (self.hook_out(output[0]),) + output[1:]
        else:
            output = self.hook_out(output)

        return output


class CLIPVisionEncoderBridge(GeneralizedComponent):
    """Bridge for the complete CLIP vision encoder.

    The CLIP vision tower consists of:
    - vision_model.embeddings: Patch + position + CLS token embeddings
    - vision_model.pre_layrnorm: LayerNorm before encoder layers
    - vision_model.encoder.layers[]: Stack of encoder layers
    - vision_model.post_layernorm: Final layer norm

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
        """Initialize the CLIP vision encoder bridge.

        Args:
            name: The name of this component (e.g., "vision_tower")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        default_submodules: Dict[str, GeneralizedComponent] = {
            "embeddings": GeneralizedComponent(name="vision_model.embeddings"),
            "pre_layernorm": NormalizationBridge(name="vision_model.pre_layrnorm", config=config),
            "encoder_layers": CLIPVisionEncoderLayerBridge(name="vision_model.encoder.layers"),
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
