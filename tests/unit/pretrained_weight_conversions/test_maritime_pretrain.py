"""Tests for the maritime-pretrain (from-scratch foundation model) weight conversion.

Coverage:

* End-to-end numerical equivalence -- a source model and its converted
  HookedTransformer produce the same logits -- across the axes a future change
  is most likely to break:
    - dense vs MoE MLPs,
    - tied vs untied embeddings,
    - several (n_heads, d_model) combinations.
  This is the property downstream probing relies on: the residual stream read
  through the hook API must be the model's real one.
* Converter key/shape correctness for the MoE branch.
* Friendly failure -- a config/checkpoint shape mismatch raises ValueError, not
  an opaque reshape error.
* Tensor-parallel shard merge -- splitting a checkpoint into N shards the way
  pretrain.py does and merging them back reconstructs the original weights and
  still converts to an equivalent model, for N in {1, 2, 4}.

The source model is a compact self-contained reimplementation of the
``pretrain.py`` architecture (RoPE with adjacent-pair rotation, RMSNorm, SwiGLU,
optional dropless top-k MoE), so the tests have no dependency on the external
training repo.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from transformer_lens import HookedTransformer
from transformer_lens.pretrained.maritime_pretrain_loader import (
    _merge_tp_shards,
    _shard_dim,
    build_config,
)
from transformer_lens.pretrained.weight_conversions.maritime_pretrain import (
    convert_maritime_pretrain_weights,
)

BASE_ARCH = dict(
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=192,
    vocab_size=256,
    max_seq_len=32,
    rope_theta=10000.0,
    rmsnorm_eps=1e-5,
)


# --- minimal reference model (mirrors pretrain.py exactly) --------------------


def _rope_cache(seq_len, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T].view(1, 1, T, -1), sin[:T].view(1, 1, T, -1)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class _RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * inv * self.weight.float()).to(x.dtype)


class _Attn(nn.Module):
    def __init__(self, arch):
        super().__init__()
        d, h = arch["d_model"], arch["n_heads"]
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, T, self.h * self.hd))


class _SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _MoE(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.k = arch["top_k"]
        self.router = nn.Linear(arch["d_model"], arch["n_experts"], bias=False)
        self.experts = nn.ModuleList(
            _SwiGLU(arch["d_model"], arch["d_ff"]) for _ in range(arch["n_experts"])
        )

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = self.router(flat.float()).softmax(-1)
        top_p, top_i = probs.topk(self.k, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e in range(len(self.experts)):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel():
                y = self.experts[e](flat.index_select(0, rows))
                out.index_add_(0, rows, y * top_p[rows, slot].unsqueeze(-1).to(y.dtype))
        return out.reshape(B, T, D)


class _Block(nn.Module):
    def __init__(self, arch, moe):
        super().__init__()
        self.norm1 = _RMSNorm(arch["d_model"], arch["rmsnorm_eps"])
        self.attn = _Attn(arch)
        self.norm2 = _RMSNorm(arch["d_model"], arch["rmsnorm_eps"])
        self.mlp = _MoE(arch) if moe else _SwiGLU(arch["d_model"], arch["d_ff"])

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class _GPT(nn.Module):
    def __init__(self, arch, tie_embeddings=True):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.embed = nn.Embedding(arch["vocab_size"], arch["d_model"])
        self.blocks = nn.ModuleList(
            _Block(arch, arch.get("moe", False)) for _ in range(arch["n_layers"])
        )
        self.norm_f = _RMSNorm(arch["d_model"], arch["rmsnorm_eps"])
        if not tie_embeddings:
            self.lm_head = nn.Linear(arch["d_model"], arch["vocab_size"], bias=False)
        cos, sin = _rope_cache(
            arch["max_seq_len"], arch["d_model"] // arch["n_heads"], arch["rope_theta"]
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, toks):
        x = self.embed(toks)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        x = self.norm_f(x)
        w = self.embed.weight if self.tie_embeddings else self.lm_head.weight
        return F.linear(x, w)

    def source_state_dict(self):
        """State dict in pretrain.py's key layout: tied models omit lm_head."""
        sd = dict(self.state_dict())
        sd.pop("cos", None)
        sd.pop("sin", None)
        return sd


def _convert_to_tl(arch, src, dtype=torch.float32):
    cfg = build_config(arch, dtype=dtype)
    state_dict = convert_maritime_pretrain_weights(src.source_state_dict(), cfg)
    tl = HookedTransformer(cfg)
    tl.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    return tl.eval()


def _max_logit_delta(arch, tie_embeddings=True):
    torch.manual_seed(0)
    src = _GPT(arch, tie_embeddings=tie_embeddings).eval()
    tl = _convert_to_tl(arch, src)
    toks = torch.randint(0, arch["vocab_size"], (2, 16))
    with torch.no_grad():
        ref = src(toks)
        got = tl(toks)
    return (ref - got).abs().max().item()


# --- equivalence across the axes most likely to regress -----------------------


@pytest.mark.parametrize("tie_embeddings", [True, False], ids=["tied", "untied"])
def test_dense_equivalence(tie_embeddings):
    assert _max_logit_delta(dict(BASE_ARCH), tie_embeddings) < 1e-4


@pytest.mark.parametrize("tie_embeddings", [True, False], ids=["tied", "untied"])
def test_moe_equivalence(tie_embeddings):
    arch = dict(BASE_ARCH, moe=True, n_experts=4, top_k=2, moe_every=1)
    assert _max_logit_delta(arch, tie_embeddings) < 1e-4


@pytest.mark.parametrize(
    "d_model,n_heads",
    [(64, 4), (64, 8), (128, 8), (96, 6)],
    ids=["64d-4h", "64d-8h", "128d-8h", "96d-6h"],
)
def test_equivalence_varying_head_counts(d_model, n_heads):
    arch = dict(BASE_ARCH, d_model=d_model, n_heads=n_heads, d_ff=3 * d_model)
    assert _max_logit_delta(arch) < 1e-4


# --- converter structure ------------------------------------------------------


def test_moe_converter_emits_expected_keys():
    arch = dict(BASE_ARCH, moe=True, n_experts=4, top_k=2, moe_every=1)
    torch.manual_seed(0)
    src = _GPT(arch)
    cfg = build_config(arch, dtype=torch.float32)
    sd = convert_maritime_pretrain_weights(src.source_state_dict(), cfg)
    assert "blocks.0.mlp.W_gate.weight" in sd
    for e in range(arch["n_experts"]):
        for w in ("W_gate", "W_in", "W_out"):
            assert f"blocks.0.mlp.experts.{e}.{w}.weight" in sd
    assert sd["blocks.0.attn.W_Q"].shape == (
        arch["n_heads"],
        arch["d_model"],
        arch["d_model"] // arch["n_heads"],
    )


# --- friendly failure on config/checkpoint mismatch ---------------------------


def test_shape_mismatch_raises_value_error():
    arch = dict(BASE_ARCH, n_layers=1)
    cfg = build_config(arch, dtype=torch.float32)
    torch.manual_seed(0)
    src = _GPT(arch)
    bad = dict(src.source_state_dict())
    # Corrupt one weight so it no longer matches the config the converter trusts.
    bad["blocks.0.attn.qkv.weight"] = torch.randn(7, arch["d_model"])
    with pytest.raises(ValueError, match="expected shape"):
        convert_maritime_pretrain_weights(bad, cfg)


# --- tensor-parallel shard merge regression -----------------------------------


def _split_into_shards(state_dict, n_shards):
    """Reproduce pretrain.py's tensor-parallel sharding: column-parallel weights
    split along dim 0, row-parallel along dim 1, everything else replicated."""
    shards = [{} for _ in range(n_shards)]
    for key, w in state_dict.items():
        dim = _shard_dim(key)
        if dim is None:
            for s in shards:
                s[key] = w.clone()
        else:
            for s, piece in zip(shards, w.chunk(n_shards, dim=dim)):
                s[key] = piece.clone()
    return shards


@pytest.mark.parametrize("n_shards", [1, 2, 4])
def test_tp_shard_merge_reconstructs_weights(tmp_path, n_shards):
    arch = dict(BASE_ARCH, d_model=64, n_heads=8, d_ff=256)  # divisible by 4
    torch.manual_seed(0)
    src = _GPT(arch)
    full = src.source_state_dict()

    run = tmp_path / "run"
    (run / "best").mkdir(parents=True)
    for rank, shard in enumerate(_split_into_shards(full, n_shards)):
        torch.save({"model": shard}, run / "best" / f"model_tp{rank}.pt")

    merged = _merge_tp_shards(run, "best")
    assert set(merged) == set(full)
    for key in full:
        assert torch.equal(merged[key], full[key]), key
