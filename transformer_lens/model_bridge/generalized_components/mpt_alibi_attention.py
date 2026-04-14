"""MPT ALiBi attention bridge — MPT uses ``position_bias`` kwarg + bool causal mask."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from packaging import version

from transformer_lens.model_bridge.generalized_components.alibi_joint_qkv_attention import (
    ALiBiJointQKVAttentionBridge,
)

try:
    import transformers as _transformers

    _TRANSFORMERS_V5 = version.parse(_transformers.__version__) >= version.parse("5.0.0")
except Exception:
    _TRANSFORMERS_V5 = False


def _build_mpt_alibi_tensor(num_heads: int, seq_len: int, alibi_bias_max: int = 8) -> torch.Tensor:
    """MPT ALiBi bias [num_heads, 1, seq_len] — mirrors HF's ``build_mpt_alibi_tensor``."""
    alibi = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(1, 1, 1, seq_len)
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))

    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.int64).float()
    base = base * (alibi_bias_max / num_heads_power_of_2)
    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)

    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1)[
            :, :num_heads, ...
        ]

    alibi = alibi * slopes  # [1, n_heads, 1, seq_len]
    return alibi.squeeze(0)  # [n_heads, 1, seq_len]


class MPTALiBiAttentionBridge(ALiBiJointQKVAttentionBridge):
    """ALiBi bridge for MPT: overrides ALiBi kwarg name, bias shape, mask format, and clip_qkv."""

    _clip_qkv: Optional[float] = None

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, None]:
        """2-tuple on transformers>=5, 3-tuple on <5 — MptBlock unpack arity changed in v5."""
        output, attn_weights = super().forward(*args, **kwargs)
        if _TRANSFORMERS_V5:
            return output, attn_weights
        return output, attn_weights, None

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        super().set_original_component(original_component)
        if hasattr(self, "o") and hasattr(original_component, "out_proj"):
            self.o.set_original_component(original_component.out_proj)
        clip = getattr(original_component, "clip_qkv", None)
        self._clip_qkv = float(clip) if clip is not None else None

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # clip_qkv is post-projection, pre-head-split — must happen before reshape.
        if self._clip_qkv is not None:
            q = q.clamp(min=-self._clip_qkv, max=self._clip_qkv)
            k = k.clamp(min=-self._clip_qkv, max=self._clip_qkv)
            v = v.clamp(min=-self._clip_qkv, max=self._clip_qkv)

        num_heads = self.config.n_heads if self.config else 32
        q, k, v, batch_size, seq_len, head_dim = self._reshape_qkv_to_heads(
            q, k, v, num_heads, num_heads
        )

        softmax_scale = head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

        # position_bias is [n_heads, 1, max_seq_len]; slice trailing kv_len, broadcast over batch.
        position_bias = kwargs.get("position_bias", None)
        if position_bias is not None:
            kv_len = attn_scores.shape[-1]
            pb = position_bias[:, :, -kv_len:]
            attn_scores = attn_scores + pb.unsqueeze(0)

        # MPT passes a bool 4D mask (True = masked), not an additive float mask.
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask, torch.finfo(attn_scores.dtype).min
            )

        attn_scores = self.hook_attn_scores(attn_scores)

        attn_weights = self._softmax_dropout_pattern(
            attn_scores, upcast_to_fp32=True, target_dtype=q.dtype
        )

        attn_output = torch.matmul(attn_weights, v)
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
        """Test inputs using MPT's kwarg names: position_bias (no batch dim) + bool causal mask."""
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        d_model = self.config.d_model if self.config and hasattr(self.config, "d_model") else 2048
        num_heads = self.config.n_heads if self.config and hasattr(self.config, "n_heads") else 32

        position_bias = _build_mpt_alibi_tensor(num_heads, seq_len).to(device=device, dtype=dtype)

        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
        causal_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        return {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype),
            "position_bias": position_bias,
            "attention_mask": causal_mask,
        }
