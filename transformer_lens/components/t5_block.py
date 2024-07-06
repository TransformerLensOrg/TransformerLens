from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import RMSNorm, T5Attention
from transformer_lens.factories.mlp_factory import MLPFactory
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from transformer_lens.utils import repeat_along_head_dimension


class T5Block(nn.Module):
    """
    T5 decoder Block. Uses T5Layernorm, and T5attention insted of usual ones.
    Also uses cross attention if is_decoder is True.
    """

    def __init__(self, cfg: HookedTransformerConfig, block_index: int, is_decoder: bool):
        super().__init__()
        self.cfg = cfg
        self.is_decoder = is_decoder

        self.ln1 = RMSNorm(cfg)
        self.attn = T5Attention(cfg, has_relative_attention_bias=block_index == 0)
        self.ln2 = RMSNorm(cfg)
        if self.is_decoder:
            self.cross_attn = T5Attention(cfg)
            self.ln3 = RMSNorm(cfg)
        self.mlp = MLPFactory.create_mlp(self.cfg)  # [batch, pos, n_heads]

        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]

        self.hook_attn_in = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        if self.is_decoder:
            self.hook_cross_attn_in = HookPoint()  # [batch, pos, d_model]
            self.hook_cross_attn_out = HookPoint()  # [batch, pos, d_model]
            self.hook_resid_mid_cross = HookPoint()  # [batch, pos, d_model]

        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
        encoder_additive_attention_mask: Optional[
            Float[torch.Tensor, "batch 1 1 encoder_pos"]
        ] = None,
        position_bias: Optional[Float[torch.Tensor, "1 head_index pos kv_pos"]] = None,
        encoder_hidden_states: Optional[Float[torch.Tensor, "batch encoder_pos d_model"]] = None,
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """A single Transformer block.

        Args:
            resid_pre (torch.Tensor): The residual stream - shape [batch, pos, d_model]
            encoder_hidden_states (torch.Tensor): The hidden states of the encoder for cross attention - shape [batch, encoder_pos, d_model]
            cache (HookedTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask for padded tokens. Defaults to None.

        Returns:
            _type_: _description_
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

        attn_in = resid_pre

        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(
                repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )

        if self.cfg.use_split_qkv_input:
            n_kv_heads = (
                self.cfg.n_key_value_heads
                if self.cfg.n_key_value_heads is not None
                else self.cfg.n_heads
            )
            query_input = self.hook_q_input(
                repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )
            key_input = self.hook_k_input(
                repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
            )
            value_input = self.hook_v_input(
                repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
            )
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in

        attn_out = self.hook_attn_out(
            # hook the residual stream states that are used to calculate the
            # queries, keys and values, independently.
            # Then take the layer norm of these inputs, and pass these to the attention module.
            self.attn(
                query_input=self.ln1(query_input),
                key_input=self.ln1(key_input),
                value_input=self.ln1(value_input),
                past_kv_cache_entry=past_kv_cache_entry,
                additive_attention_mask=additive_attention_mask,
                position_bias=position_bias,
            )
        )

        # [batch, pos, d_model]

        resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # [batch, pos, d_model]

        if self.is_decoder:
            cross_attn_in = (
                resid_mid
                if not self.cfg.use_attn_in
                else self.hook_cross_attn_in(resid_mid.clone())
            )

            if encoder_hidden_states is None:
                raise ValueError("Encoder hidden states must be provided for cross attention!")

            cross_attn_out = self.hook_cross_attn_out(
                self.cross_attn(
                    query_input=self.ln2(cross_attn_in),
                    key_input=encoder_hidden_states,
                    value_input=encoder_hidden_states,
                    additive_attention_mask=encoder_additive_attention_mask,
                )
            )
            resid_mid_cross = self.hook_resid_mid_cross(resid_mid + cross_attn_out)

            mlp_in = (
                resid_mid_cross
                if not self.cfg.use_hook_mlp_in
                else self.hook_mlp_in(resid_mid_cross.clone())
            )

            normalized_resid_mid = self.ln3(mlp_in)
        else:
            mlp_in = (
                resid_mid if not self.cfg.use_hook_mlp_in else self.hook_mlp_in(resid_mid.clone())
            )
            normalized_resid_mid = self.ln2(mlp_in)

        mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(mlp_in + mlp_out)  # [batch, pos, d_model]

        return resid_post
