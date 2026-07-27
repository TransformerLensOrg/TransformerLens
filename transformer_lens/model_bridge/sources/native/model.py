"""TL-native transformer for TransformerBridge — minimal, no HF/HT dependency.

Cfg-driven features: ``normalization_type`` (LN / RMS / RMSPre), ``final_rms``,
``gated_mlp``, ``attn_only``, ``n_key_value_heads`` (GQA), ``attn_scores_soft_cap``,
``output_logits_soft_cap``, ``positional_embedding_type`` (standard / rotary),
``rotary_dim`` / ``rotary_base`` / ``rope_scaling`` (linear PI, dynamic/NTK,
llama3 by-parts).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.utilities import TypedModuleList
from transformer_lens.utilities.activation_functions import apply_softcap

# gelu_new = the tanh-approximation HF GPT-2 / HT use; F.gelu(approximate="tanh")
# is the exact same formula.
_Activation = Callable[[torch.Tensor], torch.Tensor]
_ACTIVATIONS: dict[str, _Activation] = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
}


def _normalization_type(cfg: TransformerBridgeConfig) -> str | None:
    normalization_type = cfg.normalization_type
    return None if normalization_type is None else normalization_type.upper()


def _uses_rms_norm(cfg: TransformerBridgeConfig) -> bool:
    return _normalization_type(cfg) in ("RMS", "RMSPRE")


def _uses_no_norm(cfg: TransformerBridgeConfig) -> bool:
    return _normalization_type(cfg) is None


def _positional_kind(cfg: TransformerBridgeConfig) -> str:
    return (getattr(cfg, "positional_embedding_type", None) or "standard").lower()


class NativeRMSNorm(nn.Module):
    """Llama-style RMSNorm. Variance in fp32 regardless of input dtype, then
    cast back before the per-channel scale (matches HF LlamaRMSNorm)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms_inv = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_fp32 * rms_inv).to(input_dtype)
        return self.weight * normalized


def _make_norm(cfg: TransformerBridgeConfig, *, force_rms: bool = False) -> nn.Module:
    if force_rms or _uses_rms_norm(cfg):
        return NativeRMSNorm(cfg.d_model, eps=cfg.eps)
    if _uses_no_norm(cfg):
        return nn.Identity()
    return nn.LayerNorm(cfg.d_model, eps=cfg.eps)


def _uses_causal_attention(cfg: TransformerBridgeConfig) -> bool:
    return cfg.attention_dir == "causal"


def _resolve_rope_scaling(
    cfg: TransformerBridgeConfig, rotary_dim: int
) -> tuple[float, float, torch.Tensor]:
    """Returns (effective_base, position_scale, inv_freq) per cfg.rope_scaling."""
    base = float(cfg.rotary_base)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

    if not isinstance(rope_scaling, dict):
        return base, 1.0, inv_freq

    # Newer HF configs key on "rope_type"; older ones on "type".
    scale_type = str(rope_scaling.get("rope_type") or rope_scaling.get("type") or "").lower()
    factor = float(rope_scaling.get("factor", 1.0))

    if scale_type in ("", "default") or factor <= 1.0:
        return base, 1.0, inv_freq

    if scale_type == "linear":
        return base, factor, inv_freq

    if scale_type in ("dynamic", "ntk"):
        scaled_base = base * (factor ** (rotary_dim / (rotary_dim - 2)))
        new_inv_freq = 1.0 / (scaled_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        return scaled_base, 1.0, new_inv_freq

    if scale_type == "llama3":
        low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
        original_ctx = float(
            rope_scaling.get("original_max_position_embeddings")
            or rope_scaling.get("original_context_length")
            or 8192
        )
        low_wavelen = original_ctx / low_freq_factor
        high_wavelen = original_ctx / high_freq_factor
        wavelens = 2 * math.pi / inv_freq
        # Three regimes: low-freq → divide by factor; high-freq → unchanged;
        # in-between → smooth linear interpolation between the two.
        smooth = (original_ctx / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        new_inv_freq = torch.where(
            wavelens > low_wavelen,
            inv_freq / factor,
            torch.where(
                wavelens < high_wavelen,
                inv_freq,
                (1 - smooth) * inv_freq / factor + smooth * inv_freq,
            ),
        )
        return base, 1.0, new_inv_freq

    raise NotImplementedError(
        f"rope_scaling type {scale_type!r} is not supported. "
        f"Supported: 'linear', 'dynamic'/'ntk', 'llama3'."
    )


class NativeRotary(nn.Module):
    """Shared cos/sin tables for RoPE. Honors ``cfg.rope_scaling``."""

    # Declared so mypy sees the buffer dtype; register_buffer alone reports Module|Tensor.
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        rotary_dim = cfg.rotary_dim if cfg.rotary_dim is not None else cfg.d_head
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim!r}")
        self.rotary_dim = rotary_dim

        base, position_scale, inv_freq = _resolve_rope_scaling(cfg, rotary_dim)

        positions = torch.arange(cfg.n_ctx).float() / position_scale
        freqs = torch.outer(positions, inv_freq)
        # Llama/HF adjacent-pair format: each (2i, 2i+1) pair rotates together.
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.effective_base = base
        self.position_scale = position_scale

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Llama-style adjacent-pair rotation: (x0, x1) -> (-x1, x0).
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1)
        return rot.flatten(-2)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q/K of shape [batch, heads, seq, d_head].

        Named ``apply_rope`` rather than ``apply`` so ``nn.Module.apply(fn)``
        — PyTorch's recursive function-application utility used by
        ``bridge.apply(init_fn)`` — isn't shadowed.
        """
        seq = q.shape[-2]
        rd = self.rotary_dim
        if position_ids is None:
            cos = self.cos_cached[:seq].to(q.dtype)
            sin = self.sin_cached[:seq].to(q.dtype)
        else:
            # [batch, seq] -> [batch, 1, seq, rd] (head dim for broadcast).
            cos = self.cos_cached[position_ids].to(q.dtype).unsqueeze(1)
            sin = self.sin_cached[position_ids].to(q.dtype).unsqueeze(1)

        def _rope(x: torch.Tensor) -> torch.Tensor:
            x_rot, x_pass = x[..., :rd], x[..., rd:]
            x_rot = x_rot * cos + self._rotate_half(x_rot) * sin
            return torch.cat([x_rot, x_pass], dim=-1) if x_pass.shape[-1] else x_rot

        return _rope(q), _rope(k)


class NativeAttention(nn.Module):
    """Split-QKV causal self-attention. Returns (out, pattern); AttentionBridge
    fires ``hook_pattern`` off the second element."""

    causal_mask: torch.Tensor

    def __init__(self, cfg: TransformerBridgeConfig, rotary: Optional[NativeRotary] = None):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        self.n_kv_heads = cfg.n_key_value_heads or cfg.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_key_value_heads "
                f"({self.n_kv_heads}) for GQA."
            )
        self.kv_repeats = self.n_heads // self.n_kv_heads

        q_dim = self.n_heads * self.d_head
        kv_dim = self.n_kv_heads * self.d_head
        self.q = nn.Linear(cfg.d_model, q_dim, bias=True)
        self.k = nn.Linear(cfg.d_model, kv_dim, bias=True)
        self.v = nn.Linear(cfg.d_model, kv_dim, bias=True)
        self.o = nn.Linear(q_dim, cfg.d_model, bias=True)

        mask = torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        # attn_scale=1.0 reads like "standard scaling" but is "divide by 1" —
        # i.e. unscaled scores, which saturate softmax for d_head>1.
        if cfg.use_attn_scale and cfg.attn_scale > 0:
            if self.d_head > 1 and math.isclose(cfg.attn_scale, 1.0, abs_tol=1e-9):
                raise ValueError(
                    f"attn_scale=1.0 with d_head={self.d_head} (>1) is unscaled "
                    f"attention; softmax will saturate. For standard scaling "
                    f"leave attn_scale at -1 (sentinel for sqrt(d_head))."
                )
            scale = cfg.attn_scale
        else:
            scale = math.sqrt(cfg.d_head)
        self.scale = scale
        self.rotary = rotary
        self.attn_scores_soft_cap = float(cfg.attn_scores_soft_cap)
        self.causal = _uses_causal_attention(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        q = self.q(hidden_states).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(hidden_states).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v(hidden_states).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary.apply_rope(q, k, position_ids=position_ids)

        # GQA: repeat_interleave matches HF Llama's repeat_kv group ordering.
        if self.kv_repeats > 1:
            k = k.repeat_interleave(self.kv_repeats, dim=1)
            v = v.repeat_interleave(self.kv_repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Gemma2 soft-cap before the causal mask so masked positions stay -inf.
        scores = apply_softcap(scores, self.attn_scores_soft_cap)

        if self.causal:
            block_mask = self.causal_mask[:seq, :seq]
        else:
            block_mask = torch.zeros(seq, seq, dtype=torch.bool, device=scores.device)
        if attention_mask is not None:
            block_mask = self._combine_attention_mask(block_mask, attention_mask, batch=batch)
        scores = scores.masked_fill(block_mask, float("-inf"))

        pattern = F.softmax(scores, dim=-1)

        attn = torch.matmul(pattern, v).transpose(1, 2).contiguous().view(batch, seq, -1)
        out = self.o(attn)
        return out, pattern

    @staticmethod
    def _combine_attention_mask(
        block_mask: torch.Tensor, attention_mask: torch.Tensor, *, batch: int
    ) -> torch.Tensor:
        """Combine an external attention_mask with the causal mask.

        Accepts 2D HF padding mask ``[batch, seq]`` (1=keep, 0=mask), 4D bool
        mask (True=mask), or 4D additive float mask (HF generation style; values
        below -1 treated as masked).
        """
        if attention_mask.dim() == 2:
            pad_mask = ~attention_mask.bool()
            return block_mask | pad_mask[:, None, None, :]
        if attention_mask.dim() == 4:
            if attention_mask.dtype is torch.bool:
                return block_mask | attention_mask
            # HF additive masks use -inf or large negatives; benign biases bounded.
            extra = attention_mask < -1.0
            return block_mask | extra
        raise ValueError(
            f"attention_mask must be 2D [batch, seq] or 4D [batch, *, seq, seq], "
            f"got shape {tuple(attention_mask.shape)}."
        )


class NativeMLP(nn.Module):
    """Two-layer MLP with configurable activation."""

    act: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        assert cfg.d_mlp is not None, "NativeModel resolves d_mlp before instantiating MLPs"
        d_mlp: int = cfg.d_mlp
        self.fc_in = nn.Linear(cfg.d_model, d_mlp, bias=True)
        self.fc_out = nn.Linear(d_mlp, cfg.d_model, bias=True)
        act_name = (cfg.act_fn or "gelu").lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(f"Unsupported act_fn={act_name!r}. Supported: {sorted(_ACTIVATIONS)}")
        self.act = _ACTIVATIONS[act_name]

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fc_out(self.act(self.fc_in(hidden_states)))


class NativeGatedMLP(nn.Module):
    """SwiGLU / ReGLU / GeGLU gated MLP (variant picked by ``cfg.act_fn``).

    Submodules ``gate`` / ``in`` / ``out`` match GatedMLPBridge's expected slots.
    """

    act: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        assert cfg.d_mlp is not None, "NativeModel resolves d_mlp before instantiating MLPs"
        d_mlp: int = cfg.d_mlp
        # Llama convention: no biases on gated MLP projections.
        self.gate = nn.Linear(cfg.d_model, d_mlp, bias=False)
        # ``in`` is a Python keyword; add_module + getattr(self, "in") works
        # because the bridge resolves LinearBridge(name="in") the same way.
        self.add_module("in", nn.Linear(cfg.d_model, d_mlp, bias=False))
        self.out = nn.Linear(d_mlp, cfg.d_model, bias=False)
        # Default to SwiGLU; mirror NativeMLP's dispatch so a typo'd act_fn
        # raises instead of silently changing the model.
        act_name = (cfg.act_fn or "silu").lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(f"Unsupported act_fn={act_name!r}. Supported: {sorted(_ACTIVATIONS)}")
        self.act = _ACTIVATIONS[act_name]

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        gate_out = self.act(self.gate(hidden_states))
        in_proj = cast(nn.Linear, getattr(self, "in"))
        up_out = in_proj(hidden_states)
        return self.out(gate_out * up_out)


class NativeBlock(nn.Module):
    """Pre-LN transformer block. Layout adapts to ``cfg.attn_only`` and
    ``cfg.gated_mlp``."""

    def __init__(self, cfg: TransformerBridgeConfig, rotary: Optional[NativeRotary] = None):
        super().__init__()
        self.cfg = cfg
        self.ln1 = _make_norm(cfg)
        self.attn = NativeAttention(cfg, rotary=rotary)
        if not cfg.attn_only:
            self.ln2 = _make_norm(cfg)
            self.mlp = NativeGatedMLP(cfg) if cfg.gated_mlp else NativeMLP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        attn_out, _pattern = self.attn(
            self.ln1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attn_out
        if not self.cfg.attn_only:
            hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))
        # Tuple return matches HF block convention; BlockBridge's parser expects it.
        return (hidden_states,)


class NativeModel(nn.Module):
    """TL-native transformer. See module docstring for the supported feature set."""

    pos: Optional[nn.Embedding]
    rotary: Optional[NativeRotary]

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        # Write the resolved d_mlp back so downstream consumers see the real
        # value, not None. Mutates cfg; isolating callers should deep-copy first.
        if not getattr(cfg, "d_mlp", None):
            cfg.d_mlp = 4 * cfg.d_model
        self.cfg = cfg

        self.tok_embed = nn.Embedding(cfg.d_vocab, cfg.d_model)

        kind = _positional_kind(cfg)
        if kind == "standard":
            self.pos = nn.Embedding(cfg.n_ctx, cfg.d_model)
            self.rotary = None
        elif kind == "rotary":
            self.pos = None
            self.rotary = NativeRotary(cfg)
        else:
            raise ValueError(
                f"Unsupported positional_embedding_type={kind!r}. "
                f"NativeModel supports 'standard' and 'rotary'."
            )

        self.layers = TypedModuleList(
            [NativeBlock(cfg, rotary=self.rotary) for _ in range(cfg.n_layers)]
        )
        # final_rms forces RMS on the final norm regardless of block-norm choice
        # — matches the TL config semantic Llama uses.
        self.ln_out = _make_norm(cfg, force_rms=cfg.final_rms)
        d_vocab_out = cfg.d_vocab_out if cfg.d_vocab_out > 0 else cfg.d_vocab
        self.head = nn.Linear(cfg.d_model, d_vocab_out, bias=False)
        self.output_logits_soft_cap = float(cfg.output_logits_soft_cap)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns logits directly."""
        # Bounds check up front so both absolute and rotary paths produce a
        # self-explanatory error rather than IndexError / shape mismatch.
        seq_len = input_ids.shape[-1]
        if seq_len > self.cfg.n_ctx:
            raise ValueError(
                f"input length {seq_len} exceeds n_ctx={self.cfg.n_ctx}; "
                f"position embeddings and rotary tables are pre-baked at n_ctx."
            )

        # Resolve position_ids before the block loop so rotary sees the caller's
        # positions, not the dense default.
        batch, seq = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        hidden_states = self.tok_embed(input_ids)
        if self.pos is not None:
            hidden_states = hidden_states + self.pos(position_ids)

        for block in self.layers:
            (hidden_states,) = block(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids
            )
        hidden_states = self.ln_out(hidden_states)
        logits = self.head(hidden_states)
        logits = apply_softcap(logits, self.output_logits_soft_cap)
        return logits
