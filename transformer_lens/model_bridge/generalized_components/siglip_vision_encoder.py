"""SigLIP Vision Encoder bridge component.

This module contains the bridge component for SigLIP vision encoder layers
used in multimodal models like Gemma 3 and MedGemma.
"""
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge
from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


def _vision_attention_config(config: Any) -> Any:
    """A config view carrying the vision tower's dims, not the language model's.

    AttentionBridge reshapes its q/k/v/z hooks by ``config.n_heads`` at fire time,
    so handing it the language config reshapes vision activations by the wrong head
    count -- granite-docling runs 9 text heads over 576 dims against the tower's 12
    over 768. Reads there are hasattr-guarded, so the dims are all this needs.
    """
    n_heads = getattr(config, "vision_num_heads", None)
    d_model = getattr(config, "vision_hidden_size", None)
    if not n_heads or not d_model:
        return config
    return SimpleNamespace(n_heads=n_heads, d_model=d_model, d_head=d_model // n_heads)


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
        default_submodules: Dict[str, GeneralizedComponent] = {
            "ln1": NormalizationBridge(name="layer_norm1", config=config),
            "attn": AttentionBridge(
                name="self_attn",
                config=_vision_attention_config(config),
                submodules={
                    "q": LinearBridge(name="q_proj"),
                    "k": LinearBridge(name="k_proj"),
                    "v": LinearBridge(name="v_proj"),
                    # SigLIP names the output projection out_proj, not o_proj.
                    "o": LinearBridge(name="out_proj"),
                },
            ),
            "ln2": NormalizationBridge(name="layer_norm2", config=config),
            "mlp": MLPBridge(
                name="mlp",
                config=config,
                submodules={
                    "in": LinearBridge(name="fc1"),
                    "out": LinearBridge(name="fc2"),
                },
            ),
        }
        if submodules:
            default_submodules.update(submodules)
        super().__init__(name, config, submodules=default_submodules)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the vision encoder layer.

        Args:
            hidden_states: Input hidden states from previous layer
            attention_mask: Optional attention mask
            **kwargs: Additional arguments

        Returns:
            Output hidden states
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, attention_mask=attention_mask, **kwargs)

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
            name: The name of this component (e.g., "model.vision_tower")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        # All submodule names are resolved relative to the parent's
        # original_component (a SiglipVisionModel) by setup_submodules().
        # SiglipVisionModel wraps a SiglipVisionTransformer as .vision_model till
        # transformers version 5.6.0
        # post_layernorm is nn.LayerNorm; NormalizationBridge introspects the
        # wrapped module so the RMSNorm-LM config (Gemma 3, LLaVA) doesn't leak.
        default_submodules = {
            "embeddings": GeneralizedComponent(name="vision_model.embeddings"),
            # Pass config down: without it the layer's attention bridge inherits the
            # raw HF vision config, which spells its head count num_attention_heads
            # and so reads as n_heads=1 when the q/k/v/z hooks reshape.
            "encoder_layers": SiglipVisionEncoderLayerBridge(
                name="vision_model.encoder.layers", config=config
            ),
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
    ) -> Any:
        """Forward pass through the vision encoder.

        Args:
            pixel_values: Input image tensor [batch, channels, height, width]
            **kwargs: Additional arguments

        Returns:
            Whatever the wrapped module returns, hooked in place: a
            ``BaseModelOutput``, a tuple, or a bare tensor of vision embeddings
            [batch, num_patches, hidden_size]. Callers see HF's own shape, so the
            return is deliberately not narrowed to a tensor.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        pixel_values = self.hook_in(pixel_values)

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

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps.
        Note that SiglipVisionModel used to wrap a inner object as .vision_model before
        transformers version 5.6.0, but after that it must directly be used.
        This is a temporary hack to fix that till the transformers version is bumped.

        Args:
            original_component: The original transformer component to wrap
        """
        if not hasattr(original_component, "vision_model"):
            # We should bypass any pytorch module registration.
            object.__setattr__(original_component, "vision_model", original_component)
        super().set_original_component(original_component)
