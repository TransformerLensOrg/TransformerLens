"""fold_ln reaches the attention norm on InternLM2 and Baichuan.

Both ship a fused QKV projection (interleaved ``wqkv`` / concatenated ``W_pack``),
and both adapters used to declare ``supports_fold_ln = False`` and fold by hand in
``preprocess_weights``, keyed on ``blocks.{i}.attn.qkv.weight``. The joint-QKV
attention bridge registers a state-dict hook that deletes every ``attn.qkv.*`` key,
so that key never arrived and only ln2 and ln_final were ever folded: attention ran
in an unfolded basis while the MLP ran in a folded one. Logits stayed right, so the
mixed basis showed up only in DLA / logit-lens / factored-matrix reads.

The models here are synthetic module trees, not checkpoints -- InternLM2 and
Baichuan both load through ``trust_remote_code``, so there is no HF config class to
build a tiny model from.
"""

import math

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.baichuan import (
    BaichuanArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.internlm2 import (
    InternLM2ArchitectureAdapter,
)

D_MODEL = 32
N_HEADS = 4
N_KV_HEADS = 2
N_LAYERS = 2
D_VOCAB = 64
D_MLP = 48
HEAD_DIM = D_MODEL // N_HEADS
TOKENS = torch.tensor([[1, 5, 9, 13, 17, 21]])


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * h).to(x.dtype)


class _Rotary(nn.Module):
    """Modern HF rotary: ``forward(x, position_ids) -> (cos, sin)``."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple:
        inv_freq = self.get_buffer("inv_freq")
        freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class _LegacyRotary(nn.Module):
    """Baichuan v1/v2 rotary: ``forward(x, seq_len)`` off a ``[1, 1, seq, dim]`` cache."""

    def __init__(self, head_dim: int, max_seq: int = 64, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.einsum("i,j->ij", torch.arange(max_seq).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None]
        self.sin_cached = emb.sin()[None, None]

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> tuple:
        return (
            self.cos_cached[:, :, :seq_len].to(x.dtype),
            self.sin_cached[:, :, :seq_len].to(x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _attend(q, k, v, n_kv_heads):
    """Eager causal attention over ``[batch, seq, n_heads * head_dim]`` inputs."""
    batch, seq, _ = q.shape
    q = q.view(batch, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(batch, seq, n_kv_heads, HEAD_DIM).transpose(1, 2)
    v = v.view(batch, seq, n_kv_heads, HEAD_DIM).transpose(1, 2)
    repeats = N_HEADS // n_kv_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    scores = (q @ k.transpose(2, 3)) / math.sqrt(HEAD_DIM)
    scores = scores + torch.full((seq, seq), float("-inf")).triu(1)
    weights = scores.softmax(-1)
    out = (weights @ v).transpose(1, 2).reshape(batch, seq, N_HEADS * HEAD_DIM)
    return out, weights


class _GatedMLP(nn.Module):
    def __init__(self, gate: str, up: str, down: str) -> None:
        super().__init__()
        self._names = (gate, up, down)
        setattr(self, gate, nn.Linear(D_MODEL, D_MLP, bias=False))
        setattr(self, up, nn.Linear(D_MODEL, D_MLP, bias=False))
        setattr(self, down, nn.Linear(D_MLP, D_MODEL, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up, down = (getattr(self, name) for name in self._names)
        return down(torch.nn.functional.silu(gate(x)) * up(x))


class _InternLM2Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wqkv = nn.Linear(D_MODEL, (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
        self.rotary_emb = _Rotary(HEAD_DIM)

    def forward(self, hidden_states=None, position_ids=None, **kwargs):
        batch, seq, _ = hidden_states.shape
        groups = N_HEADS // N_KV_HEADS
        packed = self.wqkv(hidden_states).view(batch, seq, N_KV_HEADS, groups + 2, HEAD_DIM)
        q = packed[:, :, :, :groups].reshape(batch, seq, N_HEADS * HEAD_DIM)
        k = packed[:, :, :, groups].reshape(batch, seq, N_KV_HEADS * HEAD_DIM)
        v = packed[:, :, :, groups + 1].reshape(batch, seq, N_KV_HEADS * HEAD_DIM)
        out, weights = _attend(q, k, v, N_KV_HEADS)
        return self.wo(out), weights, None


class _InternLM2Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _InternLM2Attention()
        self.feed_forward = _GatedMLP("w1", "w3", "w2")
        self.attention_norm = _RMSNorm(D_MODEL)
        self.ffn_norm = _RMSNorm(D_MODEL)

    def forward(self, hidden_states, position_ids=None, **kwargs):
        attn_out, _, _ = self.attention(
            hidden_states=self.attention_norm(hidden_states), position_ids=position_ids
        )
        hidden_states = hidden_states + attn_out
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class _BaichuanAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_pack = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.rotary_emb = _LegacyRotary(HEAD_DIM)

    def forward(self, hidden_states=None, position_ids=None, **kwargs):
        packed = self.W_pack(hidden_states)
        q, k, v = packed.split(D_MODEL, dim=-1)
        out, weights = _attend(q, k, v, N_HEADS)
        return self.o_proj(out), weights, None


class _BaichuanLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _BaichuanAttention()
        self.mlp = _GatedMLP("gate_proj", "up_proj", "down_proj")
        self.input_layernorm = _RMSNorm(D_MODEL)
        self.post_attention_layernorm = _RMSNorm(D_MODEL)

    def forward(self, hidden_states, position_ids=None, **kwargs):
        attn_out, _, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states), position_ids=position_ids
        )
        hidden_states = hidden_states + attn_out
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class _Decoder(nn.Module):
    def __init__(self, embed_name: str, layer_cls) -> None:
        super().__init__()
        setattr(self, embed_name, nn.Embedding(D_VOCAB, D_MODEL))
        self._embed_name = embed_name
        self.layers = nn.ModuleList(layer_cls() for _ in range(N_LAYERS))
        self.norm = _RMSNorm(D_MODEL)

    def forward(self, input_ids, position_ids=None, **kwargs):
        hidden = getattr(self, self._embed_name)(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        for layer in self.layers:
            hidden = layer(hidden, position_ids=position_ids)
        return self.norm(hidden)


class _CausalLM(nn.Module):
    def __init__(self, embed_name: str, layer_cls, head_name: str) -> None:
        super().__init__()
        self.config = PretrainedConfig()
        self.model = _Decoder(embed_name, layer_cls)
        setattr(self, head_name, nn.Linear(D_MODEL, D_VOCAB, bias=False))
        self._head_name = head_name

    def forward(self, input_ids=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(logits=getattr(self, self._head_name)(self.model(input_ids)))


class _Tokenizer:
    pass


def _bridge_config(architecture: str, n_kv_heads: int) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=HEAD_DIM,
        n_layers=N_LAYERS,
        n_ctx=64,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        act_fn="silu",
        normalization_type="RMS",
        n_key_value_heads=n_kv_heads,
        architecture=architecture,
        model_name="synthetic",
        default_prepend_bos=True,
    )


def _randomize(model: nn.Module) -> None:
    """Non-unit norm gains, and projections wide enough for a bad fold to show.

    Left at their init values the norms are all 1.0, which makes folding a vacuous
    multiply, and the logits sit in a range so narrow that a misplaced fold moves
    them less than the tolerance.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                param.copy_(torch.rand_like(param) + 0.5)
            elif param.dim() >= 2:
                param.normal_(0.0, 0.2)


def _build(kind: str) -> TransformerBridge:
    torch.manual_seed(0)
    adapter: ArchitectureAdapter
    if kind == "internlm2":
        model = _CausalLM("tok_embeddings", _InternLM2Layer, "output")
        config = _bridge_config("InternLM2ForCausalLM", N_KV_HEADS)
        adapter = InternLM2ArchitectureAdapter(config)
    else:
        model = _CausalLM("embed_tokens", _BaichuanLayer, "lm_head")
        config = _bridge_config("BaichuanForCausalLM", N_HEADS)
        adapter = BaichuanArchitectureAdapter(config)
    _randomize(model.eval())
    return TransformerBridge(model, adapter, tokenizer=_Tokenizer())


@pytest.fixture(params=["internlm2", "baichuan"])
def bridge(request) -> TransformerBridge:
    return _build(request.param)


def _unwrap(module):
    while hasattr(module, "_original_component"):
        module = module._original_component
    return module


def _fold(bridge: TransformerBridge) -> None:
    """Fold layer norms only, so any logit movement is the fold's own doing."""
    bridge.process_weights(
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )


def _norms(bridge: TransformerBridge) -> dict[str, torch.Tensor]:
    norms = {
        f"blocks.{i}.{name}": _unwrap(getattr(block, name)).weight.detach().clone()
        for i, block in enumerate(bridge.blocks)
        for name in ("ln1", "ln2")
    }
    norms["ln_final"] = _unwrap(bridge.ln_final).weight.detach().clone()
    return norms


def _at_identity(norms: dict[str, torch.Tensor]) -> list[str]:
    return [k for k, w in norms.items() if torch.allclose(w, torch.ones_like(w), atol=1e-6)]


def _projections(bridge: TransformerBridge, layer: int) -> dict[str, torch.Tensor]:
    attn = bridge.blocks[layer].attn
    return {
        name: _unwrap(getattr(attn, name)).weight.detach().clone() for name in ("q", "k", "v", "o")
    }


def test_logits_invariant(bridge: TransformerBridge) -> None:
    with torch.no_grad():
        before = bridge(TOKENS).clone()
    # A fold misplaced onto the wrong projection has to move logits by more than the
    # tolerance for this to mean anything, so keep them off zero.
    assert before.abs().max() > 1.0
    _fold(bridge)
    with torch.no_grad():
        after = bridge(TOKENS)
    torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)


def test_every_norm_reaches_identity(bridge: TransformerBridge) -> None:
    norms = _norms(bridge)
    assert _at_identity(norms) == []
    _fold(bridge)
    folded = _norms(bridge)
    assert sorted(_at_identity(folded)) == sorted(folded)


def test_attention_norm_gain_lands_in_q_k_and_v(bridge: TransformerBridge) -> None:
    gain = _norms(bridge)["blocks.0.ln1"]
    before = _projections(bridge, 0)
    _fold(bridge)
    after = _projections(bridge, 0)
    for name in ("q", "k", "v"):
        torch.testing.assert_close(after[name], before[name] * gain[None, :], atol=1e-5, rtol=1e-5)
    # o reads the attention result, not the norm -- folding into it would double-count.
    torch.testing.assert_close(after["o"], before["o"], atol=0.0, rtol=0.0)


def test_refolding_is_a_no_op(bridge: TransformerBridge) -> None:
    _fold(bridge)
    with torch.no_grad():
        once = bridge(TOKENS).clone()
    projections = _projections(bridge, 0)
    _fold(bridge)
    with torch.no_grad():
        twice = bridge(TOKENS)
    torch.testing.assert_close(twice, once, atol=1e-4, rtol=1e-4)
    for name, weight in _projections(bridge, 0).items():
        torch.testing.assert_close(weight, projections[name], atol=1e-6, rtol=1e-6)
    assert sorted(_at_identity(_norms(bridge))) == sorted(_norms(bridge))


def test_fold_ln_false_leaves_norms_alone(bridge: TransformerBridge) -> None:
    before = _norms(bridge)
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    for key, weight in _norms(bridge).items():
        torch.testing.assert_close(weight, before[key], atol=0.0, rtol=0.0)
