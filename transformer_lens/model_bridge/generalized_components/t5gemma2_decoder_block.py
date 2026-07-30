"""T5Gemma2-specific decoder block bridge.

T5Gemma2DecoderLayer replaces T5Gemma's separate self-attention + cross-attention
with a single T5Gemma2MergedAttention module. That module computes decoder self
queries/keys/values from ``hidden_states`` and cross keys/values from
``encoder_hidden_states`` using the *same* q/k/v/o projections, concatenates the
self and cross key/value states, and runs a single softmax. As a result the
decoder layer has no separate cross-attention module and no cross-attention
layernorms — its structure mirrors the encoder layer plus encoder-state input.

This bridge monkey-patches the layer forward to fire hook points at the canonical
HookedTransformer residual-stream positions while delegating all attention math
(QK-norm, RoPE, scaling, merged KV) to the native HF module.
"""
from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class T5Gemma2DecoderBlockBridge(GeneralizedComponent):
    """Bridge for T5Gemma2 decoder layers (merged self+cross attention).

    Inserts hook points around the two sub-components of each decoder layer:
    - hook_in (hook_resid_pre): residual before self-attention pre-norm
    - hook_resid_mid: residual after merged-attention + residual add, before MLP pre-norm
    - hook_out (hook_resid_post): residual after MLP + residual add
    """

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
    ):
        super().__init__(name, config, submodules=submodules or {})
        self.hook_resid_mid = HookPoint()
        self._register_hook("hook_resid_mid", self.hook_resid_mid)
        self._original_block_forward: Optional[Callable[..., Any]] = None

    def set_original_component(self, component: torch.nn.Module) -> None:
        super().set_original_component(component)
        self._patch_decoder_layer_forward()

    def _patch_decoder_layer_forward(self) -> None:
        """Monkey-patch T5Gemma2DecoderLayer.forward to insert hook points.

        The patched forward preserves the original residual-stream semantics but
        fires hook_in, hook_resid_mid, and hook_out at the canonical
        HookedTransformer positions. Attention math stays native.
        """
        if self.original_component is None:
            return
        self._original_block_forward = self.original_component.forward

        hook_in = self.hook_in  # fires at hook_resid_pre
        hook_resid_mid = self.hook_resid_mid
        hook_out = self.hook_out  # fires at hook_resid_post

        def patched_forward(
            layer_self,
            hidden_states: torch.Tensor,
            position_embeddings=None,
            merged_attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            encoder_hidden_states=None,
            **kwargs: Any,
        ) -> torch.Tensor:
            hidden_states = hook_in(hidden_states)

            # --- merged self/cross attention sub-layer ---
            residual = hidden_states
            hidden_states = layer_self.pre_self_attn_layernorm(hidden_states)
            attn_out, _, _ = layer_self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                merged_attention_mask=merged_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_self.post_self_attn_layernorm(attn_out)
            hidden_states = residual + layer_self.dropout(hidden_states)
            hidden_states = hook_resid_mid(hidden_states)

            # --- MLP sub-layer ---
            residual = hidden_states
            hidden_states = layer_self.pre_feedforward_layernorm(hidden_states)
            hidden_states = layer_self.mlp(hidden_states)
            hidden_states = layer_self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + layer_self.dropout(hidden_states)
            hidden_states = hook_out(hidden_states)

            return hidden_states

        self.original_component.forward = types.MethodType(patched_forward, self.original_component)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )
        return self.original_component(*args, **kwargs)

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        param_names = []
        for sub_name, sub_component in self.submodules.items():
            sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
            param_names.extend(sub_component.get_expected_parameter_names(sub_prefix))
        return param_names

    def get_list_size(self) -> int:
        if self.config is None:
            return 0
        return getattr(self.config, "n_layers", 0)
