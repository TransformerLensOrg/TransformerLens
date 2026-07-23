"""T5Gemma-specific decoder block bridge.

T5GemmaDecoderLayer uses Gemma-style flat attribute access (not T5's .layer[] indexing).
It has: self-attention + cross-attention + MLP, each with pre/post norms.
This bridge monkey-patches the layer forward to insert intermediate hook points.
"""
from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class T5GemmaDecoderBlockBridge(GeneralizedComponent):
    """Bridge for T5Gemma decoder layers.

    Inserts hook points between the three sub-components of each decoder layer:
    - hook_in (hook_resid_pre): residual before self-attention pre-norm
    - hook_resid_mid: residual after self-attention + residual add, before cross-attn pre-norm
    - hook_resid_mid2: residual after cross-attention + residual add, before MLP pre-norm
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
        self.hook_resid_mid2 = HookPoint()
        self._register_hook("hook_resid_mid2", self.hook_resid_mid2)
        self._original_block_forward: Optional[Callable[..., Any]] = None

    def set_original_component(self, component: torch.nn.Module) -> None:
        super().set_original_component(component)
        self._patch_decoder_layer_forward()

    def _patch_decoder_layer_forward(self) -> None:
        """Monkey-patch T5GemmaDecoderLayer.forward to insert hook points.

        The patched forward preserves the original residual-stream semantics but
        fires hook_in, hook_resid_mid, hook_resid_mid2, and hook_out at the
        canonical HookedTransformer positions.
        """
        if self.original_component is None:
            return
        self._original_block_forward = self.original_component.forward

        hook_in = self.hook_in  # fires at hook_resid_pre
        hook_resid_mid = self.hook_resid_mid
        hook_resid_mid2 = self.hook_resid_mid2
        hook_out = self.hook_out  # fires at hook_resid_post
        original_forward = self._original_block_forward

        def patched_forward(
            layer_self,
            hidden_states: torch.Tensor,
            position_embeddings=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            **kwargs: Any,
        ) -> torch.Tensor:
            hidden_states = hook_in(hidden_states)

            # --- self-attention sub-layer ---
            residual = hidden_states
            hidden_states = layer_self.pre_self_attn_layernorm(hidden_states)
            sa_out, _ = layer_self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=(
                    past_key_values.self_attention_cache if past_key_values is not None else None
                ),
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_self.post_self_attn_layernorm(sa_out)
            hidden_states = residual + layer_self.dropout(hidden_states)
            hidden_states = hook_resid_mid(hidden_states)

            # --- cross-attention sub-layer ---
            residual = hidden_states
            hidden_states = layer_self.pre_cross_attn_layernorm(hidden_states)
            ca_out, _ = layer_self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_self.post_cross_attn_layernorm(ca_out)
            hidden_states = residual + layer_self.dropout(hidden_states)
            hidden_states = hook_resid_mid2(hidden_states)

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
