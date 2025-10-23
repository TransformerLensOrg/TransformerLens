"""T5-specific block bridge component.

This module contains the bridge component for T5 blocks, which have a different
structure than standard transformer blocks (3 layers in decoder vs 2 layers).
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class T5BlockBridge(GeneralizedComponent):
    """Bridge component for T5 transformer blocks.

    T5 has two types of blocks:
    - Encoder blocks: 2 layers (self-attention, feed-forward)
    - Decoder blocks: 3 layers (self-attention, cross-attention, feed-forward)

    This bridge handles both types based on the presence of cross-attention.
    """

    # Override the class attribute to indicate this is a list item
    is_list_item: bool = True

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        is_decoder: bool = False,
    ):
        """Initialize the T5 block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration
            submodules: Dictionary of submodules to register
            is_decoder: Whether this is a decoder block (has cross-attention)
        """
        super().__init__(name, config, submodules=submodules or {})
        self.is_decoder = is_decoder

        # Create hook points for residual streams
        self.hook_resid_mid = HookPoint()  # After self-attention
        self._register_hook("hook_resid_mid", self.hook_resid_mid)

        if is_decoder:
            # Decoder has an additional residual point after cross-attention
            self.hook_resid_mid2 = HookPoint()  # After cross-attention
            self._register_hook("hook_resid_mid2", self.hook_resid_mid2)

        self._original_block_forward: Optional[Callable[..., Any]] = None

    def set_original_component(self, component: torch.nn.Module):
        """Set the original component and monkey-patch its forward method.

        Args:
            component: The original PyTorch module to wrap
        """
        super().set_original_component(component)

        # Monkey-patch the block's forward method to insert hooks
        self._patch_t5_block_forward()

    def _patch_t5_block_forward(self):
        """Monkey-patch the T5 block's forward method to insert hooks."""
        if self.original_component is None:
            return

        # Store the original forward method
        self._original_block_forward = self.original_component.forward

        # Create new forward method that inserts hooks
        def patched_forward(
            block_self,  # This is the T5 block instance
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
            **kwargs,  # Catch any additional arguments like cache_position
        ):
            """Patched T5 block forward with hooks."""

            # Apply hook_in (hook_resid_pre)
            hidden_states = self.hook_in(hidden_states)

            # Get the layer list from the T5 block
            # T5 blocks have a "layer" attribute which is a ModuleList
            if not hasattr(block_self, "layer"):
                raise RuntimeError(f"T5 block {block_self} does not have 'layer' attribute")

            layers = block_self.layer

            # Determine block type based on number of layers
            is_decoder_block = len(layers) == 3

            # Layer 0: Self-Attention
            if past_key_value is not None:
                if not is_decoder_block:
                    # Encoder doesn't use past_key_value
                    expected_num_past_key_values = 0
                else:
                    # Decoder: 2 for self-attention, 2 for cross-attention
                    expected_num_past_key_values = 2

                if len(past_key_value) != expected_num_past_key_values:
                    raise ValueError(
                        f"There should be {expected_num_past_key_values} past states. "
                        f"Got {len(past_key_value)}."
                    )

                self_attn_past_key_value = past_key_value[:2] if is_decoder_block else None
                cross_attn_past_key_value = past_key_value[2:4] if is_decoder_block else None
            else:
                self_attn_past_key_value = None
                cross_attn_past_key_value = None

            # Self-attention layer
            self_attention_outputs = layers[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=self_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = self_attention_outputs[0]
            present_key_value_state = self_attention_outputs[1] if use_cache else None

            # Apply hook after self-attention
            hidden_states = self.hook_resid_mid(hidden_states)

            # Cross-attention (decoder only)
            if is_decoder_block and encoder_hidden_states is not None:
                # Cross-attention is layer[1] in decoder blocks
                cross_attention_outputs = layers[1](
                    hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_bias=encoder_decoder_position_bias,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                hidden_states = cross_attention_outputs[0]

                # Apply hook after cross-attention
                if hasattr(self, "hook_resid_mid2"):
                    hidden_states = self.hook_resid_mid2(hidden_states)

                # Append cross-attention KV cache if using cache
                if use_cache:
                    present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Feed-forward layer
            # Layer index is 1 for encoder, 2 for decoder
            ff_layer_idx = 2 if is_decoder_block else 1
            feed_forward_outputs = layers[ff_layer_idx](hidden_states)
            hidden_states = feed_forward_outputs[0]

            # Apply hook_out (hook_resid_post)
            hidden_states = self.hook_out(hidden_states)

            # Build outputs - use tuple concatenation to handle variable-length tuples
            outputs: tuple[Any, ...] = (hidden_states,)

            if use_cache:
                outputs = outputs + (present_key_value_state,)

            if output_attentions:
                outputs = outputs + (self_attention_outputs[2],)  # Self-attention weights
                if is_decoder_block and encoder_hidden_states is not None:
                    outputs = outputs + (cross_attention_outputs[2],)  # Cross-attention weights

            return outputs

        # Replace the forward method
        self.original_component.forward = types.MethodType(patched_forward, self.original_component)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the block bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            The output from the original component
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        # Hooks are applied inside the patched forward method
        output = self.original_component(*args, **kwargs)

        return output

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this block.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        param_names = []

        # Delegate to all subcomponents
        for sub_name, sub_component in self.submodules.items():
            sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
            param_names.extend(sub_component.get_expected_parameter_names(sub_prefix))

        return param_names

    def get_list_size(self) -> int:
        """Get the number of transformer blocks.

        Returns:
            Number of layers in the model
        """
        if self.config is None:
            return 0
        # For T5, encoder and decoder have same number of layers
        return getattr(self.config, "n_layers", 0)
