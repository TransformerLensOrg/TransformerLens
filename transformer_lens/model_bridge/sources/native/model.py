"""TL-native transformer model for use with TransformerBridge.

A minimal, from-scratch transformer implementation with no HuggingFace or
HookedTransformer dependency. Internal attribute names are deliberately chosen
to NOT collide with the bridge's top-level component slot names
("embed", "blocks", "ln_final", "unembed") — the bridge's __getattr__ falls back
to ``original_model.<slot>`` and an HF-style collision would block add_module
during bridge setup.

Features driven by config fields:

- ``normalization_type``: ``"LN"`` (default) or ``"RMS"`` / ``"RMSPre"``.
- ``final_rms``: when True, the final norm uses RMS regardless of block norm.
- ``gated_mlp``: when True, swaps in a SwiGLU-style gated MLP (Llama/Mistral).
- ``attn_only``: when True, blocks have no MLP / no ln2.
- ``n_key_value_heads``: when set and < ``n_heads``, enables grouped-query
  attention (Llama 3.x / Mistral / DeepSeek style).
- ``attn_scores_soft_cap``: when > 0, applies Gemma2-style tanh soft-cap to
  pre-softmax attention scores.
- ``output_logits_soft_cap``: when > 0, applies tanh soft-cap to final logits.
- ``positional_embedding_type``: ``"standard"`` (absolute, default) or
  ``"rotary"``. Rotary applies inside attention; absolute uses ``self.pos``.
- ``rotary_dim``: partial-rotary dim (rotates first ``rotary_dim`` of each
  head; pass-through the rest). Default ``d_head``.
- ``rotary_base``: RoPE base frequency. Default ``10000``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.config import TransformerBridgeConfig

# gelu_new is the tanh-approximation of GELU (what HF GPT-2 and HookedTransformer
# use). PyTorch's F.gelu accepts approximate="tanh" since 1.10 — that's exactly
# the same formula, no need to roll our own.
_ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
}


def _uses_rms_norm(cfg: TransformerBridgeConfig) -> bool:
    return (cfg.normalization_type or "LN").upper() in ("RMS", "RMSPRE")


def _positional_kind(cfg: TransformerBridgeConfig) -> str:
    return (getattr(cfg, "positional_embedding_type", None) or "standard").lower()


class NativeRMSNorm(nn.Module):
    """Root-mean-square LayerNorm. No mean centering, no bias.

    Matches the math used by Llama / Mistral / T5: ``y = w * x / rms(x)`` where
    ``rms(x) = sqrt(mean(x^2) + eps)``. The variance is computed in fp32
    regardless of input dtype — mirroring HF Llama's LlamaRMSNorm — so bf16/fp16
    inputs don't accumulate variance drift. The result is cast back to the
    input dtype before the per-channel scale, so the scale runs in the user's
    chosen precision.

    The bridge's RMSNormalizationBridge wraps any module with a ``weight``
    attribute and a forward returning the normalized tensor — no further
    coordination required.
    """

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
    return nn.LayerNorm(cfg.d_model, eps=cfg.eps)


class NativeRotary(nn.Module):
    """Pre-computes the cos/sin tables used by RoPE.

    Lives at the model level (one shared instance) so all attention layers
    re-use the same buffers. Per-call, we just slice to the current sequence
    length. No HF dependency.
    """

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        rotary_dim = cfg.rotary_dim if cfg.rotary_dim is not None else cfg.d_head
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim!r}")
        self.rotary_dim = rotary_dim
        base = float(cfg.rotary_base)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        positions = torch.arange(cfg.n_ctx).float()
        freqs = torch.outer(positions, inv_freq)  # [n_ctx, rotary_dim/2]
        # Adjacent-pair format (the form Llama/HF use): each pair (2i, 2i+1)
        # rotates together. We expand cos/sin per element of each pair.
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # [n_ctx, rotary_dim]
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Llama-style adjacent-pair rotation: (x0, x1) -> (-x1, x0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1)
        return rot.flatten(-2)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K. Tensors are [batch, heads, seq, d_head]."""
        seq = q.shape[-2]
        rd = self.rotary_dim
        cos = self.cos_cached[:seq].to(q.dtype)  # [seq, rd]
        sin = self.sin_cached[:seq].to(q.dtype)

        def _rope(x: torch.Tensor) -> torch.Tensor:
            x_rot, x_pass = x[..., :rd], x[..., rd:]
            x_rot = x_rot * cos + self._rotate_half(x_rot) * sin
            return torch.cat([x_rot, x_pass], dim=-1) if x_pass.shape[-1] else x_rot

        return _rope(q), _rope(k)


class NativeAttention(nn.Module):
    """Split-QKV causal self-attention with optional GQA, RoPE, and soft-cap.

    Returns ``(attn_output, attention_weights)`` so the bridge's AttentionBridge
    fires ``hook_pattern`` off the second element.
    """

    def __init__(self, cfg: TransformerBridgeConfig, rotary: Optional[NativeRotary] = None):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        # n_key_value_heads governs GQA: K/V have fewer heads than Q. Default
        # to n_heads (= standard multi-head attention).
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

        scale = (
            cfg.attn_scale if cfg.use_attn_scale and cfg.attn_scale > 0 else math.sqrt(cfg.d_head)
        )
        self.scale = scale
        self.rotary = rotary  # None unless cfg.positional_embedding_type == "rotary"
        self.attn_scores_soft_cap = float(cfg.attn_scores_soft_cap)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, _ = hidden_states.shape

        q = self.q(hidden_states).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(hidden_states).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v(hidden_states).view(batch, seq, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary.apply(q, k)

        # Expand K/V to match Q head count under GQA. repeat_interleave keeps
        # group ordering consistent with HF Llama's repeat_kv.
        if self.kv_repeats > 1:
            k = k.repeat_interleave(self.kv_repeats, dim=1)
            v = v.repeat_interleave(self.kv_repeats, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Gemma2-style attention soft-cap: c * tanh(scores / c). Bounds raw
        # logits before the causal mask so masked positions stay -inf.
        if self.attn_scores_soft_cap > 0:
            c = self.attn_scores_soft_cap
            scores = c * torch.tanh(scores / c)
        mask = self.causal_mask[:seq, :seq]
        scores = scores.masked_fill(mask, float("-inf"))
        pattern = F.softmax(scores, dim=-1)

        attn = torch.matmul(pattern, v).transpose(1, 2).contiguous().view(batch, seq, -1)
        out = self.o(attn)
        return out, pattern


class NativeMLP(nn.Module):
    """Two-layer MLP with configurable activation."""

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        d_mlp = cfg.d_mlp
        self.fc_in = nn.Linear(cfg.d_model, d_mlp, bias=True)
        self.fc_out = nn.Linear(d_mlp, cfg.d_model, bias=True)
        act_name = (cfg.act_fn or "gelu").lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(f"Unsupported act_fn={act_name!r}. Supported: {sorted(_ACTIVATIONS)}")
        self.act = _ACTIVATIONS[act_name]

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fc_out(self.act(self.fc_in(hidden_states)))


class NativeGatedMLP(nn.Module):
    """SwiGLU-style gated MLP (Llama / Mistral / Gemma2).

    Submodule names ``gate`` / ``in`` / ``out`` align with the bridge's
    GatedMLPBridge submodule slots; the adapter wires them by these names.
    """

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        d_mlp = cfg.d_mlp
        # No biases by default — matches Llama. Users wanting biased gated MLPs
        # can subclass; toy-scope stays simple.
        self.gate = nn.Linear(cfg.d_model, d_mlp, bias=False)
        # ``in`` is a Python keyword, so we can't write ``self.in = ...`` —
        # but ``add_module`` accepts any string and stores it in ``_modules``,
        # so ``getattr(self, "in")`` resolves it the same way the bridge does
        # when walking ``LinearBridge(name="in")``. No __getattr__ override
        # required.
        self.add_module("in", nn.Linear(cfg.d_model, d_mlp, bias=False))
        self.out = nn.Linear(d_mlp, cfg.d_model, bias=False)
        # Gated MLPs typically pair with SiLU/swish; honor cfg if the user picked
        # a different activation, but default to silu.
        act_name = (cfg.act_fn or "silu").lower()
        if act_name == "gelu":  # GeGLU variant
            self.act = _ACTIVATIONS["gelu"]
        elif act_name == "gelu_new":
            self.act = _ACTIVATIONS["gelu_new"]
        else:
            self.act = F.silu

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        gate_out = self.act(self.gate(hidden_states))
        up_out = getattr(self, "in")(hidden_states)
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

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        attn_out, _pattern = self.attn(self.ln1(hidden_states))
        hidden_states = hidden_states + attn_out
        if not self.cfg.attn_only:
            hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))
        # Tuple return matches HF block convention so BlockBridge's parser is happy.
        return (hidden_states,)


class NativeModel(nn.Module):
    """TL-native transformer. See module docstring for the supported feature set."""

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__()
        # Resolve defaults that NativeMLP / nn.Embedding need, and write them
        # back so downstream consumers reading cfg.d_mlp see the real value
        # instead of None. Mutates the supplied cfg; callers that want isolation
        # (e.g. TransformerBridge.boot_native) deep-copy the user's cfg before
        # constructing the model.
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

        self.layers = nn.ModuleList(
            [NativeBlock(cfg, rotary=self.rotary) for _ in range(cfg.n_layers)]
        )
        # final_rms overrides the block-norm choice — Llama uses LN-equivalent
        # blocks but final_rms is true in TL config to opt into RMSNorm on the
        # final norm. We honor the same semantic.
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
        """Returns logits directly. The bridge unwraps either .logits, tuple[0],
        or a bare tensor — we pick the simplest path.
        """
        hidden_states = self.tok_embed(input_ids)
        if self.pos is not None:
            batch, seq = input_ids.shape
            if position_ids is None:
                position_ids = (
                    torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
                )
            hidden_states = hidden_states + self.pos(position_ids)

        for block in self.layers:
            (hidden_states,) = block(hidden_states)
        hidden_states = self.ln_out(hidden_states)
        logits = self.head(hidden_states)
        # Gemma2-style output soft-cap.
        if self.output_logits_soft_cap > 0:
            c = self.output_logits_soft_cap
            logits = c * torch.tanh(logits / c)
        return logits
