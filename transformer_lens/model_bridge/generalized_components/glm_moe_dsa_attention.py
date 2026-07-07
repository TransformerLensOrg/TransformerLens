"""GLM-MoE-DSA attention bridge component."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.mla_attention import (
    MLAAttentionBridge,
)


def _apply_rotary_pos_emb_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int
) -> torch.Tensor:
    """Apply interleaved-pair rotary position embeddings (transformers >= 5.13).

    HF 5.13 switched GLM-MoE-DSA from split-half NeoX-style RoPE to interleaved-
    pair rotation (``apply_rotary_pos_emb_interleave``). Even-dimension elements
    are paired with the following odd dimension: (d0,d1), (d2,d3), …
    """
    cos = cos[..., :cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., :sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GlmMoeDsaAttentionBridge(MLAAttentionBridge):
    """Bridge for GLM-5 DeepSeek Sparse Attention.

    GLM-MoE-DSA extends MLA with a learned top-k token indexer and returns
    ``(attn_output, attn_weights, topk_indices_or_none)`` to feed shared
    top-k indices into later layers.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, config, submodules=submodules, **kwargs)
        self.hook_topk_indices = HookPoint()
        self.hook_dsa_mask = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        hf_attn: Any = self.original_component

        if not self._mla_params_initialized:
            self._q_lora_rank = getattr(hf_attn, "q_lora_rank", None)
            self._kv_lora_rank = getattr(hf_attn, "kv_lora_rank")
            self._qk_nope_head_dim = getattr(hf_attn, "qk_nope_head_dim")
            self._qk_rope_head_dim = getattr(hf_attn, "qk_rope_head_dim")
            self._v_head_dim = getattr(hf_attn, "v_head_dim")
            self._qk_head_dim = getattr(
                hf_attn, "qk_head_dim", self._qk_nope_head_dim + self._qk_rope_head_dim
            )
            self._n_heads = getattr(hf_attn, "num_heads")
            self._mla_params_initialized = True

        if "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
            args = args[1:]
        else:
            raise ValueError("Could not find hidden_states in args or kwargs")

        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)
        past_key_values = kwargs.pop("past_key_values", None)
        prev_topk_indices = kwargs.pop("prev_topk_indices", None)
        position_ids = kwargs.pop("position_ids", None)

        hidden_states = self.hook_in(hidden_states)
        batch_size, seq_length = hidden_states.shape[:-1]

        if self._q_lora_rank is None:
            query_states = hf_attn.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = hf_attn.q_a_layernorm(hf_attn.q_a_proj(hidden_states))
            q_resid = self.hook_q_latent(q_resid)
            query_states = hf_attn.q_b_proj(q_resid)

        query_states = query_states.view(batch_size, seq_length, -1, self._qk_head_dim).transpose(
            1, 2
        )
        q_nope, q_pe = torch.split(
            query_states, [self._qk_nope_head_dim, self._qk_rope_head_dim], dim=-1
        )

        compressed_kv = hf_attn.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(
            compressed_kv, [self._kv_lora_rank, self._qk_rope_head_dim], dim=-1
        )
        k_compressed = hf_attn.kv_a_layernorm(k_compressed)
        k_compressed = self.hook_kv_latent(k_compressed)

        kv_expanded = hf_attn.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(
            batch_size, seq_length, -1, self._qk_nope_head_dim + self._v_head_dim
        )
        k_nope, value_states = torch.split(
            kv_expanded, [self._qk_nope_head_dim, self._v_head_dim], dim=-1
        )
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            position_embeddings = self._apply_position_embedding_hooks(position_embeddings)
            cos, sin = position_embeddings
        elif self._rotary_emb is not None:
            position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            cos, sin = self._rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)
        else:
            raise ValueError(
                "GlmMoeDsaAttentionBridge requires position_embeddings or set_rotary_emb()."
            )

        q_pe = _apply_rotary_pos_emb_single(q_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.view(batch_size, 1, seq_length, self._qk_rope_head_dim)
        k_pe = _apply_rotary_pos_emb_single(k_pe, cos, sin, unsqueeze_dim=1)
        q_pe = self.hook_rot_q(q_pe)
        k_pe = self.hook_rot_k(k_pe)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)
        query_states = self.hook_q(query_states)
        key_states = self.hook_k(key_states)
        value_states = self.hook_v(value_states)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, hf_attn.layer_idx
            )

        if hf_attn.indexer is not None:
            if attention_mask is not None and attention_mask.dim() == 4:
                indexer_mask = attention_mask[:, 0, :, :]
            elif attention_mask is not None:
                indexer_mask = attention_mask.unsqueeze(1)
            else:
                indexer_mask = None
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            topk_indices = hf_attn.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                position_ids,
                past_key_values=past_key_values,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer (prev_topk_indices is None)."
                )
            topk_indices = prev_topk_indices
        topk_indices = self.hook_topk_indices(topk_indices)

        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)
        index_mask = self.hook_dsa_mask(index_mask).unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            attn_scores_mask = index_mask + attention_mask[..., :total_len]
        elif attention_mask is not None:
            attn_scores_mask = attention_mask.masked_fill(
                index_mask == float("-inf"), float("-inf")
            )
        else:
            causal_mask = torch.arange(
                total_len, device=hidden_states.device
            )[None, None, None, :] > torch.arange(
                q_pe.shape[-2], device=hidden_states.device
            )[:, None, None]
            index_mask = index_mask.masked_fill(causal_mask, float("-inf"))
            attn_scores_mask = index_mask

        key_states = _repeat_kv(key_states, hf_attn.num_key_value_groups)
        value_states = _repeat_kv(value_states, hf_attn.num_key_value_groups)
        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * hf_attn.scaling
        attn_scores = attn_scores + attn_scores_mask
        attn_scores = self.hook_attn_scores(attn_scores)
        attn_weights = self._softmax_dropout_pattern(
            attn_scores, upcast_to_fp32=True, target_dtype=query_states.dtype
        )
        if self.training and hf_attn.attention_dropout:
            attn_weights = F.dropout(attn_weights, p=hf_attn.attention_dropout, training=True)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = hf_attn.o_proj(attn_output)
        attn_output = self.hook_out(attn_output)
        return attn_output, attn_weights, topk_indices
