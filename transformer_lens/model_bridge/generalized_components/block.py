"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component provides standardized input/output hooks and monkey-patches
    HuggingFace blocks to insert hooks at positions matching HookedTransformer.
    """

    # Override the class attribute to indicate this is a list item
    is_list_item: bool = True

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        # hook_resid_mid is handled specially via monkey-patching (after attn, before ln2)
        "hook_resid_post": "hook_out",
        "hook_attn_in": "attn.hook_in",
        "hook_attn_out": "attn.hook_out",
        "hook_q_input": "attn.q.hook_in",
        "hook_k_input": "attn.k.hook_in",
        "hook_v_input": "attn.v.hook_in",
        "hook_mlp_in": "mlp.hook_in",
        "hook_mlp_out": "mlp.hook_out",  # Alias hook_mlp_out to mlp.hook_out
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for BlockBridge)
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules)

        # Create custom hook_resid_mid that will be inserted via monkey-patching
        # This hook captures the residual stream after attention but before ln2
        # Unlike the alias to ln2.hook_in, this ensures gradients don't pass through LayerNorm
        self.hook_resid_mid = HookPoint()
        self._register_hook("hook_resid_mid", self.hook_resid_mid)

        self._original_block_forward: Optional[Callable[..., Any]] = None

    def set_original_component(self, component: torch.nn.Module):
        """Set the original component and monkey-patch its forward method.

        This method monkey-patches HuggingFace blocks to insert hook_mlp_out
        at the correct position (after MLP, before residual addition), matching
        HookedTransformer's architecture.

        Args:
            component: The original PyTorch module to wrap
        """
        super().set_original_component(component)

        # Monkey-patch the block's forward method to insert hook_mlp_out
        self._patch_block_forward()

    def _patch_block_forward(self):
        """Monkey-patch the HuggingFace block's forward method.

        This inserts hook_mlp_out between the MLP and the residual addition,
        matching HookedTransformer's architecture where hook_mlp_out sees
        gradients before the residual split.
        """
        if self.original_component is None:
            return

        # Store the original forward method
        self._original_block_forward = self.original_component.forward

        # Create new forward method that inserts hook_mlp_out
        def patched_forward(
            block_self,  # This is the HF block instance
            hidden_states,
            past_key_value=None,
            cache_position=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            position_embeddings=None,  # Gemma2 and other models pass position_embeddings
            **kwargs,
        ):
            # Call original forward but intercept MLP output
            # Architecture-agnostic: supports GPT-2, GPT-NeoX, OPT, etc.

            # Apply hook_in (hook_resid_pre) at the start, matching HookedTransformer
            hidden_states = self.hook_in(hidden_states)

            # Attention block
            residual = hidden_states

            # Get architecture-specific attention name (attn, attention, self_attn, etc.)
            attn = (
                getattr(block_self, "attn", None)
                or getattr(block_self, "attention", None)
                or getattr(block_self, "self_attn", None)
            )
            if attn is None:
                raise RuntimeError(f"Could not find attention module in block {block_self}")

            # Check if attention expects pre-ln1 input (for split Q/K/V compatibility with HookedTransformer)
            # When enabled, attention will call ln1 three separate times internally
            expects_pre_ln1 = getattr(attn, "_expects_pre_ln1_input", False)

            if expects_pre_ln1:
                # Attention will handle ln1 internally (3 separate calls for Q, K, V)
                attn_input = residual
            else:
                # Normal path: apply ln1 once here in the block
                ln1 = (
                    getattr(block_self, "ln_1", None)
                    or getattr(block_self, "input_layernorm", None)
                    or getattr(block_self, "self_attn_layer_norm", None)
                )
                if ln1 is not None:
                    hidden_states = ln1(hidden_states)
                attn_input = hidden_states

            # Some models use different parameter names for KV cache (e.g., GPTNeo uses 'layer_past')
            # Detect which parameter name the original HF attention expects
            import inspect

            # Check the original HF attention if the attention is wrapped
            check_attn = getattr(attn, "original_component", attn)
            attn_sig = inspect.signature(
                check_attn.forward if hasattr(check_attn, "forward") else check_attn.__call__
            )
            attn_params = set(attn_sig.parameters.keys())

            attn_kwargs = {
                "cache_position": cache_position,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
                **kwargs,
            }

            # Handle position_embeddings for models like Gemma2
            # Position embeddings need to be passed through to attention
            if position_embeddings is not None:
                attn_kwargs["position_embeddings"] = position_embeddings

            # Add KV cache with the correct parameter name
            if past_key_value is not None:
                if "layer_past" in attn_params:
                    attn_kwargs["layer_past"] = past_key_value
                elif "past_key_value" in attn_params:
                    attn_kwargs["past_key_value"] = past_key_value
                else:
                    # Fallback: if neither is found explicitly,
                    # use past_key_value as the default (most common)
                    attn_kwargs["past_key_value"] = past_key_value

            attn_result = attn(attn_input, **attn_kwargs)  # type: ignore[misc]
            # Handle different return formats: (output, weights) or (output, weights, past)
            if len(attn_result) >= 2:
                attn_output = attn_result[0]
                attn_weights = attn_result[1]
            else:
                attn_output = attn_result
                attn_weights = None
            # Residual connection
            hidden_states = attn_output + residual

            # Apply hook_resid_mid (after attention, before ln2)
            # This matches HookedTransformer where hook_resid_mid is separate from ln2
            hidden_states = self.hook_resid_mid(hidden_states)

            # Cross attention (if applicable)
            if encoder_hidden_states is not None:
                if not hasattr(block_self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {block_self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                hidden_states = block_self.ln_cross_attn(hidden_states)
                cross_attn_output, cross_attn_weights = block_self.crossattention(
                    hidden_states,
                    past_key_value=past_key_value,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                # Residual connection
                hidden_states = residual + cross_attn_output

            # MLP block - THIS IS WHERE WE INSERT hook_mlp_out
            residual = hidden_states
            # Get architecture-specific second layer norm name (ln_2, post_attention_layernorm, final_layer_norm, etc.)
            ln2 = (
                getattr(block_self, "ln_2", None)
                or getattr(block_self, "post_attention_layernorm", None)
                or getattr(block_self, "final_layer_norm", None)
            )
            if ln2 is not None:
                hidden_states = ln2(hidden_states)

            # Get architecture-specific MLP name (mlp, fc1+fc2, etc.)
            mlp = getattr(block_self, "mlp", None)
            if mlp is not None:
                feed_forward_hidden_states = mlp(hidden_states)
            else:
                # OPT uses fc1 and fc2 instead of a combined mlp module
                fc1 = getattr(block_self, "fc1", None)
                fc2 = getattr(block_self, "fc2", None)
                if fc1 is not None and fc2 is not None:
                    import torch.nn.functional as F

                    hidden_states = fc1(hidden_states)
                    hidden_states = F.relu(hidden_states)  # OPT uses ReLU
                    feed_forward_hidden_states = fc2(hidden_states)
                else:
                    raise RuntimeError(f"Could not find MLP module in block {block_self}")

            # Residual connection
            hidden_states = residual + feed_forward_hidden_states

            # Apply hook_resid_post (hook_out) INSIDE the block, matching HT architecture
            # This is critical for correct gradient flow!
            hidden_states = self.hook_out(hidden_states)

            outputs: tuple[Any, ...] = (hidden_states,)
            if output_attentions:
                outputs = outputs + (attn_weights,)
                if encoder_hidden_states is not None:
                    outputs = outputs + (cross_attn_weights,)

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
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # NOTE: hook_in and hook_out are now applied INSIDE the patched forward
        # method to match HookedTransformer's architecture. We don't apply them
        # here in the wrapper to avoid double-wrapping.
        output = self.original_component(*args, **kwargs)

        # If output is a single-element tuple, unwrap it
        # This prevents tuples from being passed between blocks
        if isinstance(output, tuple) and len(output) == 1:
            return output[0]

        return output

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this block component.

        Block components delegate to their subcomponents to get parameter names.

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

        For BlockBridge, this returns n_layers from the config.

        Returns:
            Number of layers in the model
        """
        if self.config is None:
            return 0
        return getattr(self.config, "n_layers", 0)
