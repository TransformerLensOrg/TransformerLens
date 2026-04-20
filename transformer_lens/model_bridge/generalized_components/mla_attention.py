"""Multi-Head Latent Attention (MLA) bridge component for DeepSeek models.

MLA compresses Q and KV into lower-dimensional latent spaces via LoRA-style
projections before standard attention. This component reimplements the MLA
forward path step-by-step with hooks at each meaningful stage, exposing:

- hook_q_latent / hook_kv_latent: compressed representations (the information bottleneck)
- hook_q / hook_k / hook_v: final Q/K/V entering attention (post-decompression, post-RoPE)
- hook_rot_q / hook_rot_k: after RoPE on the rope portion splits
- hook_attn_scores / hook_pattern: pre/post-softmax attention weights
- hook_z: pre-output-projection (alias for o.hook_in)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.position_embedding_hooks_mixin import (
    PositionEmbeddingHooksMixin,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims of the input (standard RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k tensors."""
    cos = cos.unsqueeze(1)  # [batch, 1, seq, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class MLAAttentionBridge(PositionEmbeddingHooksMixin, AttentionBridge):
    """Bridge for DeepSeek's Multi-Head Latent Attention (MLA).

    Reimplements the MLA forward path with hooks at each computation stage.
    Standard W_Q/W_K/W_V properties are not available on MLA models — use
    the submodule weight access (q_a_proj, q_b_proj, etc.) instead.
    """

    # MLA has no standard q/k/v submodules — override to empty
    property_aliases: Dict[str, str] = {}

    hook_aliases = {
        "hook_result": "hook_out",
        "hook_z": "o.hook_in",
    }

    def __init__(
        self,
        name: str,
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, config, submodules=submodules, **kwargs)
        self._init_position_embedding_hooks()

        self.hook_q_latent = HookPoint()  # Compressed Q (post q_a_layernorm)
        self.hook_kv_latent = HookPoint()  # Compressed KV (post kv_a_layernorm)
        self.hook_q = HookPoint()  # Final Q entering attention (post-RoPE concat)
        self.hook_k = HookPoint()  # Final K entering attention (post-RoPE concat)
        self.hook_v = HookPoint()  # V from kv_b_proj split
        self.hook_rot_q = HookPoint()  # Q rope portion after RoPE
        self.hook_rot_k = HookPoint()  # K rope portion after RoPE

        # MLA params lazy-initialized from HF module (bridge config lacks these fields)
        self._mla_params_initialized = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Reimplemented MLA forward with hooks at each computation stage.

        Follows the DeepseekV3Attention forward path, calling into HF submodules
        individually and firing hooks at each meaningful stage.
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        hf_attn: Any = self.original_component

        if not self._mla_params_initialized:
            self._q_lora_rank = getattr(hf_attn, "q_lora_rank", None)
            self._kv_lora_rank = getattr(hf_attn, "kv_lora_rank", 512)
            self._qk_nope_head_dim = getattr(hf_attn, "qk_nope_head_dim", 128)
            self._qk_rope_head_dim = getattr(hf_attn, "qk_rope_head_dim", 64)
            self._v_head_dim = getattr(hf_attn, "v_head_dim", 128)
            self._qk_head_dim = self._qk_nope_head_dim + self._qk_rope_head_dim
            self._n_heads = getattr(hf_attn, "num_heads", 32)
            hf_config = getattr(hf_attn, "config", None)
            self._rope_interleave = (
                getattr(hf_config, "rope_interleave", False) if hf_config else False
            )
            self._mla_params_initialized = True

        # --- Extract inputs ---
        if "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
            args = args[1:]
        else:
            raise ValueError("Could not find hidden_states in args or kwargs")

        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)

        hidden_states = self.hook_in(hidden_states)

        batch_size, seq_length = hidden_states.shape[:2]

        # --- Query path ---
        if self._q_lora_rank is None:
            # Direct projection (no compression)
            q_states = hf_attn.q_proj(hidden_states)
        else:
            # Two-stage compression: q_a_proj → q_a_layernorm → q_b_proj
            q_compressed = hf_attn.q_a_proj(hidden_states)
            q_compressed = hf_attn.q_a_layernorm(q_compressed)
            q_compressed = self.hook_q_latent(q_compressed)
            q_states = hf_attn.q_b_proj(q_compressed)

        # Reshape to [batch, n_heads, seq, qk_head_dim]
        q_states = q_states.view(batch_size, seq_length, -1, self._qk_head_dim).transpose(1, 2)
        # Split into nope (non-RoPE) and pe (RoPE) portions
        q_pass, q_rot = torch.split(
            q_states, [self._qk_nope_head_dim, self._qk_rope_head_dim], dim=-1
        )

        # --- KV path ---
        # kv_a_proj_with_mqa outputs [compressed_kv || k_pe]
        compressed_kv_full = hf_attn.kv_a_proj_with_mqa(hidden_states)
        # Split: compressed KV latent (for kv_b_proj) and k rope portion (for direct RoPE)
        # Note: k_pe is split off here and goes directly to RoPE — hook_kv_latent
        # captures only the compressed_kv portion that enters the decompression path.
        k_pass, k_rot = torch.split(
            compressed_kv_full, [self._kv_lora_rank, self._qk_rope_head_dim], dim=-1
        )

        # Compress → normalize → decompress the KV latent
        k_pass = hf_attn.kv_a_layernorm(k_pass)
        k_pass = self.hook_kv_latent(k_pass)
        k_pass = hf_attn.kv_b_proj(k_pass)

        # Reshape to [batch, n_heads, seq, nope+v_head]
        key_shape = (batch_size, seq_length, -1, self._qk_nope_head_dim + self._v_head_dim)
        k_pass = k_pass.view(key_shape).transpose(1, 2)
        # Split K nope portion and V
        k_pass, value_states = torch.split(
            k_pass, [self._qk_nope_head_dim, self._v_head_dim], dim=-1
        )

        # k_rot is [batch, seq, rope_dim] → [batch, 1, seq, rope_dim] for broadcasting
        k_rot = k_rot.view(batch_size, 1, seq_length, self._qk_rope_head_dim)

        # --- RoPE ---
        if position_embeddings is not None:
            position_embeddings = self._apply_position_embedding_hooks(position_embeddings)
            cos, sin = position_embeddings
        elif self._rotary_emb is not None:
            # Fallback: compute from rotary_emb if position_embeddings not passed
            position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0)
            cos, sin = self._rotary_emb(hidden_states, position_ids)
        else:
            raise ValueError(
                "MLAAttentionBridge requires position_embeddings or set_rotary_emb() "
                "to be called before forward."
            )

        q_rot, k_rot = _apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        q_rot = self.hook_rot_q(q_rot)
        k_rot = self.hook_rot_k(k_rot)

        # Expand k_rot to match the number of heads
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        # Concatenate nope + rope portions to form final Q and K
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # Fire final Q/K/V hooks — these are the tensors entering attention
        query_states = self.hook_q(query_states)
        key_states = self.hook_k(key_states)
        value_states = self.hook_v(value_states)

        # --- KV Cache ---
        past_key_values = kwargs.pop("past_key_values", None)
        cache_position = kwargs.pop("cache_position", None)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, hf_attn.layer_idx, cache_kwargs
            )

        # --- Attention computation (no V padding — only needed for flash attention) ---
        scaling = self._qk_head_dim ** (-0.5)
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        attn_weights = self._softmax_dropout_pattern(
            attn_scores, upcast_to_fp32=True, target_dtype=query_states.dtype
        )

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)

        # --- Output projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = hf_attn.o_proj(attn_output)

        attn_output = self.hook_out(attn_output)
        return attn_output, attn_weights

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Generate test inputs with hidden_states, position_embeddings, and attention_mask."""
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        # Try bridge config (d_model), then HF attention's config (hidden_size), then fallback
        d_model = None
        if self.config and hasattr(self.config, "d_model"):
            d_model = self.config.d_model
        if d_model is None and self.original_component is not None:
            hf_cfg = getattr(self.original_component, "config", None)
            if hf_cfg is not None:
                d_model = getattr(hf_cfg, "hidden_size", None)
        if d_model is None:
            d_model = 256
        inputs: Dict[str, Any] = {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
        }

        # Generate position_embeddings from rotary_emb if available,
        # otherwise create dummy (cos=1, sin=0) embeddings
        rope_head_dim = self._qk_rope_head_dim if self._mla_params_initialized else 64
        if self._rotary_emb is not None:
            try:
                dummy_input = inputs["hidden_states"]
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                position_embeddings = self._rotary_emb(dummy_input, position_ids)
                inputs["position_embeddings"] = position_embeddings
            except Exception:
                cos = torch.ones(1, seq_len, rope_head_dim, device=device, dtype=dtype)
                sin = torch.zeros(1, seq_len, rope_head_dim, device=device, dtype=dtype)
                inputs["position_embeddings"] = (cos, sin)
        else:
            cos = torch.ones(1, seq_len, rope_head_dim, device=device, dtype=dtype)
            sin = torch.zeros(1, seq_len, rope_head_dim, device=device, dtype=dtype)
            inputs["position_embeddings"] = (cos, sin)

        inputs["attention_mask"] = None
        return inputs

    def __getattr__(self, name: str) -> Any:
        """Raise clear error for standard weight properties that don't apply to MLA."""
        if name in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"):
            raise NotImplementedError(
                f"{name} is not available on MLA (Multi-Head Latent Attention) models. "
                f"MLA uses compressed projections instead of standard Q/K/V. "
                f"Access weights via submodules: q_a_proj, q_b_proj, kv_a_proj_with_mqa, "
                f"kv_b_proj, o (o_proj)."
            )
        return super().__getattr__(name)
