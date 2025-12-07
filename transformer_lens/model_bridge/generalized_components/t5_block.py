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

    is_list_item: bool = True
    hook_aliases = {"hook_resid_pre": "hook_in", "hook_resid_post": "hook_out"}

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
        self.hook_resid_mid = HookPoint()
        self._register_hook("hook_resid_mid", self.hook_resid_mid)
        if is_decoder:
            self.hook_resid_mid2 = HookPoint()
            self._register_hook("hook_resid_mid2", self.hook_resid_mid2)
        self._original_block_forward: Optional[Callable[..., Any]] = None

    def set_original_component(self, component: torch.nn.Module):
        """Set the original component and monkey-patch its forward method.

        Args:
            component: The original PyTorch module to wrap
        """
        super().set_original_component(component)
        self._patch_t5_block_forward()

    def _patch_t5_block_forward(self):
        """Monkey-patch the T5 block's forward method to insert hooks."""
        if self.original_component is None:
            return
        self._original_block_forward = self.original_component.forward

        def patched_forward(
            block_self,
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
            cache_position=None,
            **kwargs,
        ):
            """Patched T5 block forward with hooks."""
            hidden_states = self.hook_in(hidden_states)
            if not hasattr(block_self, "layer"):
                raise RuntimeError(f"T5 block {block_self} does not have 'layer' attribute")
            layers = block_self.layer
            is_decoder_block = len(layers) == 3
            if past_key_value is not None:
                if not is_decoder_block:
                    expected_num_past_key_values = 0
                else:
                    expected_num_past_key_values = 2
                if len(past_key_value) != expected_num_past_key_values:
                    raise ValueError(
                        f"There should be {expected_num_past_key_values} past states. Got {len(past_key_value)}."
                    )
                self_attn_past_key_value = past_key_value[:2] if is_decoder_block else None
                cross_attn_past_key_value = past_key_value[2:4] if is_decoder_block else None
            else:
                self_attn_past_key_value = None
                cross_attn_past_key_value = None
            self_attention_outputs = layers[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=self_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = self_attention_outputs[0]
            # Keep self-attention outputs and relative position weights
            # attention_outputs contains: (position_bias,) or (position_bias, attn_weights)
            attention_outputs = self_attention_outputs[1:]
            hidden_states = self.hook_resid_mid(hidden_states)
            if is_decoder_block and encoder_hidden_states is not None:
                cross_attention_outputs = layers[1](
                    hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_bias=encoder_decoder_position_bias,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )
                hidden_states = cross_attention_outputs[0]
                if hasattr(self, "hook_resid_mid2"):
                    hidden_states = self.hook_resid_mid2(hidden_states)
                # Keep cross-attention outputs and relative position weights
                attention_outputs = attention_outputs + cross_attention_outputs[1:]
            ff_layer_idx = 2 if is_decoder_block else 1
            feed_forward_outputs = layers[ff_layer_idx](hidden_states)
            # T5LayerFF returns a tensor, not a tuple
            if isinstance(feed_forward_outputs, tuple):
                hidden_states = feed_forward_outputs[0]
            else:
                hidden_states = feed_forward_outputs
            hidden_states = self.hook_out(hidden_states)
            outputs: tuple[Any, ...] = (hidden_states,)
            # Return: hidden-states, (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            return outputs + attention_outputs

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
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
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
        return getattr(self.config, "n_layers", 0)
