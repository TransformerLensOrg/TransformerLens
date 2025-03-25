import math
from typing import Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformer_lens.hook_points import HookPoint
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(p=2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        x_normed = x / (rms_x + self.eps)
        return x_normed * self.weight

class AbstractAttention(nn.Module):
    def __init__(self, cfg, attention_types, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        # Retain older naming of parameters
        self.head_dim = cfg.d_head
        self.num_heads = cfg.n_heads

        self.is_sliding = bool((layer_idx + 1) % cfg.sliding_window_pattern)
        self.sliding_window = cfg.window_size if self.is_sliding else None

        # Same param structure
        self.W_Q = nn.Parameter(torch.empty(self.num_heads, cfg.d_model, self.head_dim, dtype=cfg.dtype))
        self._W_K = nn.Parameter(torch.empty(self.num_heads, cfg.d_model, self.head_dim, dtype=cfg.dtype))
        self._W_V = nn.Parameter(torch.empty(self.num_heads, cfg.d_model, self.head_dim, dtype=cfg.dtype))
        self.W_O = nn.Parameter(torch.empty(self.num_heads, self.head_dim, cfg.d_model, dtype=cfg.dtype))

        # Biases at shape [n_heads, d_head], but we'll reshape for broadcasting
        self.b_Q = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, dtype=cfg.dtype))
        self._b_K = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, dtype=cfg.dtype))
        self._b_V = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, dtype=cfg.dtype))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))

        # RMSNorm layers for Q/K
        self.q_norm = RMSNorm(self.head_dim, cfg.eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.eps)

        # Hook points (for TransformerLens integration)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_result = HookPoint()

        # For old-school Gemma RoPE: We'll store cos/sin in buffers if needed
        self.register_buffer("cos_cached", torch.zeros(1), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(1), persistent=False)

    def _compute_sin_cos(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute old-school Gemma cos/sin for RoPE. We'll keep it simple, replicating typical logic."""
        # Example logic: the freq base is 10000, dimension is self.head_dim.
        # We'll produce shape [1, seq_len, 1, head_dim] so it can broadcast.
        # You can customize for your exact old Gemma approach.

        half_dim = self.head_dim // 2
        freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 1.0, device=device) / half_dim))
        # freq shape = [half_dim]

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("s,d->sd", t, freq)

        # Now expand to shape [seq_len, head_dim]
        freqs = torch.cat([freqs, freqs], dim=1)  # double to match full head_dim
        # final shape for broadcasting across (batch, n_heads)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        return cos, sin

    def forward(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"]
        ],
        key_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos kv_head_index d_model"]
        ],
        value_input: Union[
            Float[torch.Tensor, "batch kv_pos d_model"],
            Float[torch.Tensor, "batch kv_pos head_index d_model"],
            Float[torch.Tensor, "batch kv_pos kv_head_index d_model"]
        ],
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 kv_pos"]] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
        position_bias: Optional[Float[torch.Tensor, "1 head_index pos kv_pos"]] = None,
        **kwargs,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # ====== 1) Project Q/K/V  ======
        bQ = self.b_Q.view(1, 1, self.num_heads, self.head_dim)
        bK = self.b_K.view(1, 1, self.num_heads, self.head_dim)
        bV = self.b_V.view(1, 1, self.num_heads, self.head_dim)

        q = torch.einsum("bpd,hdf->bphf", query_input, self.W_Q) + bQ  # [batch, pos, heads, head_dim]
        k = torch.einsum("bpd,hdf->bphf", key_input,   self._W_K) + bK
        v = torch.einsum("bpd,hdf->bphf", value_input, self._W_V) + bV

        # ====== 2) RMSNorm for Q/K ======
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ====== 3) Old-school Gemma approach: compute sin/cos internally
        seq_len = q.size(1)
        cos, sin = self._compute_sin_cos(seq_len, device=q.device)

        # We'll slice cos, sin to match the actual length
        cos = cos[:, :seq_len, :, :]
        sin = sin[:, :seq_len, :, :]

        # apply the transform
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        # ====== 4) Past KV Cache: use .append(...) if available ======
        # if past_kv_cache_entry is not None:
        #     kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
        #     k, v = past_kv_cache_entry.append(k, v)

        # ====== 5) Compute attn scores [batch, head, pos, pos] ======
        attn_scores = torch.einsum("bphf,bqhf->bhpq", q, k) / (self.head_dim ** 0.5)

        if position_bias is not None:
            attn_scores = attn_scores + position_bias

        if additive_attention_mask is not None:
            attn_scores = attn_scores + additive_attention_mask

        if attention_mask is not None:
            pass

        pattern = F.softmax(attn_scores, dim=-1)
        pattern = self.hook_pattern(pattern)

        # ====== 6) Multiply pattern with values, reproject ======
        attn_output = torch.einsum("bhpq,bqhf->bphf", pattern, v)
        # attn_output = attn_output.contiguous().view(attn_output.size(0), attn_output.size(1), -1)

        # ====== 7) Output projection ======
        result = torch.einsum("bphf,hfm->bpm", attn_output, self.W_O)
        result = result + self.b_O  # [batch, pos, d_model]
        result = self.hook_result(result)

        return result

    def _apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Equivalent of apply_rotary_pos_emb from old code, inlined here."""

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
