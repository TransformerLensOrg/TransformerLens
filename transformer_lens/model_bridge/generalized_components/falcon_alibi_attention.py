"""Falcon ALiBi attention bridge component.

Handles Falcon models that use ALiBi (Attention with Linear Biases) instead of RoPE.
Splits fused QKV, reimplements attention with ALiBi bias and hooks at each stage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.alibi_utils import (
    build_alibi_tensor,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)


class FalconALiBiAttentionBridge(JointQKVAttentionBridge):
    """Attention bridge for Falcon models using ALiBi position encoding.

    Splits fused QKV, reimplements attention with ALiBi bias fused into scores,
    and fires hooks at each stage (hook_q, hook_k, hook_v, hook_attn_scores,
    hook_pattern). ALiBi bias is added to raw attention scores before scaling.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        split_qkv_matrix: Any = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            config=config,
            split_qkv_matrix=split_qkv_matrix,
            submodules=submodules,
            requires_position_embeddings=False,
            requires_attention_mask=False,
            **kwargs,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass: split QKV, apply ALiBi, fire hooks."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        else:
            raise ValueError("Could not find hidden_states in args or kwargs")

        hooked_input = self.hook_in(hidden_states)

        # Split fused QKV via parent's split mechanism
        q_output = self.q(hooked_input)
        k_output = self.k(hooked_input)
        v_output = self.v(hooked_input)

        attn_output, attn_weights = self._reconstruct_attention(
            q_output, k_output, v_output, **kwargs
        )

        output = self.hook_out(attn_output)
        return output, attn_weights

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct attention with ALiBi bias fused into scores."""
        num_heads = self.config.n_heads if self.config else 32
        num_kv_heads = getattr(self.config, "n_key_value_heads", None) or num_heads
        q, k, v, batch_size, seq_len, head_dim = self._reshape_qkv_to_heads(
            q, k, v, num_heads, num_kv_heads
        )

        # GQA/MQA: expand K/V heads to match Q heads
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        inv_norm_factor = head_dim**-0.5

        # Raw attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores.view(batch_size, num_heads, seq_len, -1)

        # Upcast for numerical stability
        input_dtype = attn_scores.dtype
        if input_dtype in (torch.float16, torch.bfloat16):
            attn_scores = attn_scores.to(torch.float32)

        # Add ALiBi bias
        alibi = kwargs.get("alibi", None)
        if alibi is not None:
            kv_len = attn_scores.shape[-1]
            alibi_view = alibi.view(batch_size, num_heads, 1, -1)
            if alibi_view.shape[-1] > kv_len:
                alibi_view = alibi_view[..., :kv_len]
            attn_scores = attn_scores + alibi_view

        # Scale after ALiBi (matches HF Falcon)
        attn_scores = attn_scores * inv_norm_factor

        # Add attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask[:, :, :, : attn_scores.shape[-1]]

        attn_scores = self.hook_attn_scores(attn_scores)

        attn_weights = self._softmax_dropout_pattern(
            attn_scores, upcast_to_fp32=True, target_dtype=q.dtype
        )

        # Weighted sum
        attn_output = torch.matmul(
            attn_weights.view(batch_size * num_heads, seq_len, -1),
            v.reshape(batch_size * num_heads, -1, head_dim),
        )
        attn_output = attn_output.view(batch_size, num_heads, seq_len, head_dim)
        attn_output = self._reshape_attn_output(
            attn_output, batch_size, seq_len, num_heads, head_dim
        )
        attn_output = self._apply_output_projection(attn_output)

        return attn_output, attn_weights

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Generate test inputs including ALiBi tensor and attention mask."""
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 2048
        num_heads = self.config.n_heads if self.config and hasattr(self.config, "n_heads") else 32

        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        # HF Falcon passes alibi as [batch*heads, 1, seq] — reshape to match
        alibi_4d = build_alibi_tensor(attention_mask, num_heads, dtype)
        alibi = alibi_4d.reshape(batch_size * num_heads, 1, seq_len)

        # Causal mask: [batch, 1, seq, seq]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        return {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype),
            "alibi": alibi,
            "attention_mask": causal_mask,
        }
