"""Offline integration tests for one dense LLaDA transformer forward pass."""

from __future__ import annotations

import copy
import gc
import math
import weakref
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, cast
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import PreTrainedTokenizerFast

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


@dataclass
class TinyLLaDAConfig:
    """Small structural copy of the released dense LLaDA config contract."""

    activation_type: str = "silu"
    alibi: bool = False
    architectures: list[str] = field(default_factory=lambda: ["LLaDAModelLM"])
    attention_dropout: float = 0.0
    attention_layer_norm: bool = False
    block_group_size: int = 1
    block_type: str = "llama"
    d_model: int = 32
    embedding_dropout: float = 0.0
    embedding_size: int = 64
    eos_token_id: int = 1
    include_bias: bool = False
    include_qkv_bias: bool = False
    input_emb_norm: bool = False
    layer_norm_type: str = "rms"
    mask_token_id: int = 63
    max_sequence_length: int = 16
    mlp_hidden_size: int = 48
    model_type: str = "llada"
    n_heads: int = 4
    n_kv_heads: int = 2
    n_layers: int = 2
    output_attentions: bool = True
    pad_token_id: int = 0
    residual_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope: bool = True
    rope_full_precision: bool = True
    rope_theta: float = 500_000.0
    scale_logits: bool = False
    use_cache: bool = False
    vocab_size: int = 64
    weight_tying: bool = False

    @property
    def effective_n_kv_heads(self) -> int:
        return self.n_kv_heads


class TinyLLaDAOutput(NamedTuple):
    logits: torch.Tensor


class TinyRMSNorm(nn.Module):
    def __init__(self, config: TinyLLaDAConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model))
        self.eps = config.rms_norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(original_dtype)


class TinyRotaryEmbedding(nn.Module):
    """LLaDA's half-vector RoPE pairing, computed in fp32."""

    def __init__(self, config: TinyLLaDAConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        batch, heads, positions, head_dim = x.shape
        paired = x.view(batch, heads, positions, 2, head_dim // 2)
        first, second = paired.unbind(dim=-2)
        return torch.cat((-second, first), dim=-1)

    def _apply_rotary(
        self, tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
    ) -> torch.Tensor:
        return (tensor * cos + self._rotate_half(tensor) * sin).to(tensor.dtype)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_query_dtype = query.dtype
        original_key_dtype = key.dtype
        if self.config.rope_full_precision:
            query = query.float()
            key = key.float()

        head_dim = query.shape[-1]
        key_len = key.shape[-2]
        query_len = query.shape[-2]
        inverse_frequency = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, head_dim, 2, device=query.device).float() / head_dim)
        )
        frequency = torch.outer(
            torch.arange(key_len, device=query.device).float(), inverse_frequency
        )
        position = torch.cat((frequency, frequency), dim=-1)
        sin = position.sin()[None, None, :, :]
        cos = position.cos()[None, None, :, :]
        query = self._apply_rotary(
            query, sin[:, :, key_len - query_len :], cos[:, :, key_len - query_len :]
        )
        key = self._apply_rotary(key, sin, cos)
        return query.to(original_query_dtype), key.to(original_key_dtype)


class TinyLLaDALlamaBlock(nn.Module):
    """Faithful tiny copy of the remote dense ``LLaDALlamaBlock`` math."""

    def __init__(self, config: TinyLLaDAConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.residual_dropout)
        self.act = nn.SiLU()
        self.rotary_emb = TinyRotaryEmbedding(config)
        self.attn_norm = TinyRMSNorm(config)
        self.ff_norm = TinyRMSNorm(config)

        head_dim = config.d_model // config.n_heads
        kv_width = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.ff_proj = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        self.ff_out = nn.Linear(config.mlp_hidden_size, config.d_model, bias=False)

        self.last_attn_norm: Optional[torch.Tensor] = None
        self.last_q: Optional[torch.Tensor] = None
        self.last_k: Optional[torch.Tensor] = None
        self.last_v: Optional[torch.Tensor] = None
        self.last_rot_q: Optional[torch.Tensor] = None
        self.last_rot_k: Optional[torch.Tensor] = None
        self.last_scores: Optional[torch.Tensor] = None
        self.last_pattern: Optional[torch.Tensor] = None
        self.last_attn_output: Optional[torch.Tensor] = None
        self.last_resid_mid: Optional[torch.Tensor] = None
        self.last_mlp_norm: Optional[torch.Tensor] = None
        self.last_gate: Optional[torch.Tensor] = None
        self.last_up: Optional[torch.Tensor] = None
        self.last_activation: Optional[torch.Tensor] = None
        self.last_product: Optional[torch.Tensor] = None
        self.last_mlp_output: Optional[torch.Tensor] = None
        self.last_output: Optional[torch.Tensor] = None

    @staticmethod
    def _cast_attn_bias(bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        return bias.to(dtype=input_dtype)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, query_len, q_width = q.shape
        head_dim = q_width // self.config.n_heads
        q = q.view(batch, query_len, self.config.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch, query_len, self.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, query_len, self.config.effective_n_kv_heads, head_dim).transpose(1, 2)

        self.last_q = q
        self.last_k = k
        self.last_v = v

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = (k, v) if use_cache else None

        q, k = self.rotary_emb(q, k)
        self.last_rot_q = q
        self.last_rot_k = k
        key_len = k.shape[-2]
        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len],
                k.dtype,
            )

        groups = self.config.n_heads // self.config.effective_n_kv_heads
        k = k.repeat_interleave(groups, dim=1, output_size=self.config.n_heads)
        v = v.repeat_interleave(groups, dim=1, output_size=self.config.n_heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attention_bias is not None:
            scores = scores + attention_bias
        self.last_scores = scores
        pattern = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        z = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        z = z.transpose(1, 2).contiguous()
        projected = self.attn_out(z.view(batch, query_len, self.config.d_model))

        self.last_pattern = pattern
        self.last_attn_output = projected
        return projected, present

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        normalized = self.attn_norm(x)
        self.last_attn_norm = normalized
        q = self.q_proj(normalized)
        k = self.k_proj(normalized)
        v = self.v_proj(normalized)
        attention_output, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        resid_mid = x + self.dropout(attention_output)

        mlp_norm = self.ff_norm(resid_mid)
        gate = self.ff_proj(mlp_norm)
        up = self.up_proj(mlp_norm)
        activation = self.act(gate)
        product = activation * up
        mlp_output = self.dropout(self.ff_out(product))
        output = resid_mid + mlp_output

        self.last_resid_mid = resid_mid
        self.last_mlp_norm = mlp_norm
        self.last_gate = gate
        self.last_up = up
        self.last_activation = activation
        self.last_product = product
        self.last_mlp_output = mlp_output
        self.last_output = output
        return output, cache


class TinyLLaDAModel(nn.Module):
    def __init__(self, config: TinyLLaDAConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.embedding_size, config.d_model),
                "emb_drop": nn.Dropout(config.embedding_dropout),
                "blocks": nn.ModuleList(
                    [TinyLLaDALlamaBlock(config) for _ in range(config.n_layers)]
                ),
                "ln_f": TinyRMSNorm(config),
                "ff_out": nn.Linear(config.d_model, config.embedding_size, bias=False),
            }
        )
        self.last_final_norm: Optional[torch.Tensor] = None

    @staticmethod
    def _padding_bias(
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None or not bool((attention_mask == 0).any()):
            return None
        mask = attention_mask.float().view(batch_size, -1)[:, None, None, :]
        return (1.0 - mask) * torch.finfo(mask.dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> TinyLLaDAOutput:
        del output_hidden_states
        if use_cache:
            raise AssertionError("The KV cache is not supported for LLaDA")

        batch_size, seq_len = input_ids.shape
        x = self.transformer["wte"](input_ids)
        x = self.transformer["emb_drop"](x)
        padding_bias = self._padding_bias(attention_mask, batch_size)
        if attention_bias is None and padding_bias is not None:
            attention_bias = torch.zeros(1, 1, seq_len, seq_len, device=x.device)
        if attention_bias is not None and attention_bias.dtype in (torch.bool, torch.int8):
            allowed = attention_bias.bool()
            attention_bias = torch.where(
                allowed,
                torch.zeros((), device=x.device),
                torch.full((), torch.finfo(torch.float32).min, device=x.device),
            )
        if attention_bias is not None and padding_bias is not None:
            attention_bias = attention_bias.float() + padding_bias

        blocks = cast(nn.ModuleList, self.transformer["blocks"])
        for block_module in blocks:
            x, _ = block_module(x, attention_bias=attention_bias, use_cache=False)
        x = self.transformer["ln_f"](x)
        self.last_final_norm = x
        logits = self.transformer["ff_out"](x)
        return TinyLLaDAOutput(logits=logits)


class TinyLLaDAModelLM(nn.Module):
    def __init__(self, config: TinyLLaDAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TinyLLaDAModel(config)
        self.output_attentions_requests: list[Optional[bool]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TinyLLaDAOutput:
        self.output_attentions_requests.append(output_attentions)
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")
        if use_cache is None:
            use_cache = self.config.use_cache
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )


class TinyModels(NamedTuple):
    bridge: TransformerBridge
    reference: TinyLLaDAModelLM


def _build_models(
    config: Optional[TinyLLaDAConfig] = None,
    *,
    dtype: torch.dtype = torch.float32,
    eval_bridge: bool = True,
) -> TinyModels:
    torch.manual_seed(0)
    config = config or TinyLLaDAConfig()
    reference = TinyLLaDAModelLM(config).to(dtype=dtype).eval()
    candidate = copy.deepcopy(reference).eval()
    bridge = build_bridge_from_module(
        candidate,
        architecture="LLaDAModelLM",
        hf_config=copy.deepcopy(candidate.config),
        tokenizer=None,
        device="cpu",
        dtype=dtype,
        model_name="offline-tiny-llada",
    )
    if eval_bridge:
        bridge.eval()
    return TinyModels(bridge=bridge, reference=reference)


@pytest.fixture(scope="module")
def models() -> TinyModels:
    return _build_models()


def _offline_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "<mask>": 63,
    }
    vocabulary.update(
        {f"token_{index}": index for index in range(64) if index not in {0, 1, 2, 3, 63}}
    )
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )


def _reference_block(reference: TinyLLaDAModelLM, index: int = 0) -> TinyLLaDALlamaBlock:
    blocks = cast(nn.ModuleList, reference.model.transformer["blocks"])
    block = blocks[index]
    assert isinstance(block, TinyLLaDALlamaBlock)
    return block


def _require_tensor(value: Optional[torch.Tensor]) -> torch.Tensor:
    assert value is not None
    return value


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_remote_code_and_revision_reach_all_hf_loaders(trust_remote_code: bool) -> None:
    import transformer_lens.model_bridge.sources.transformers as bridge_source

    config = TinyLLaDAConfig()
    model = TinyLLaDAModelLM(TinyLLaDAConfig()).eval()
    tokenizer = _offline_tokenizer()
    with (
        patch.object(
            bridge_source.AutoConfig,
            "from_pretrained",
            return_value=copy.deepcopy(config),
        ) as config_load,
        patch.object(
            bridge_source.AutoModelForCausalLM,
            "from_pretrained",
            return_value=model,
        ) as model_load,
        patch.object(
            bridge_source.AutoTokenizer,
            "from_pretrained",
            return_value=tokenizer,
        ) as tokenizer_load,
    ):
        bridge = TransformerBridge.boot_transformers(
            "offline-tiny-llada",
            device="cpu",
            trust_remote_code=trust_remote_code,
            revision="reviewed-revision",
        )

    assert bridge.cfg.trust_remote_code is trust_remote_code
    config_kwargs = config_load.call_args.kwargs
    model_kwargs = model_load.call_args.kwargs
    tokenizer_kwargs = tokenizer_load.call_args.kwargs
    assert config_kwargs["trust_remote_code"] is trust_remote_code
    assert config_kwargs["revision"] == "reviewed-revision"
    assert model_kwargs.get("trust_remote_code", False) is trust_remote_code
    assert model_kwargs["revision"] == "reviewed-revision"
    assert tokenizer_kwargs["trust_remote_code"] is trust_remote_code
    assert tokenizer_kwargs["revision"] == "reviewed-revision"


def test_masked_forward_matches_reference_logits_and_intermediates(
    models: TinyModels,
) -> None:
    tokens = torch.tensor([[5, 63, 7, 9]])
    with torch.inference_mode():
        reference_logits = models.reference(tokens).logits
        bridge_logits, cache = models.bridge.run_with_cache(tokens)

    reference_block = _reference_block(models.reference)
    torch.testing.assert_close(bridge_logits, reference_logits, rtol=1e-5, atol=1e-6)
    attn_norm = _require_tensor(reference_block.last_attn_norm)
    torch.testing.assert_close(cache["blocks.0.ln1.hook_out"], attn_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cache["blocks.0.attn.hook_in"], attn_norm, rtol=1e-5, atol=1e-6)

    reference_q = _require_tensor(reference_block.last_q)
    reference_k = _require_tensor(reference_block.last_k)
    reference_v = _require_tensor(reference_block.last_v)
    torch.testing.assert_close(
        cache["blocks.0.attn.q.hook_out"],
        reference_q.transpose(1, 2),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.k.hook_out"],
        reference_k.transpose(1, 2),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.v.hook_out"],
        reference_v.transpose(1, 2),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.hook_rot_q"],
        _require_tensor(reference_block.last_rot_q).transpose(1, 2),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.hook_rot_k"],
        _require_tensor(reference_block.last_rot_k).transpose(1, 2),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.hook_attn_scores"],
        _require_tensor(reference_block.last_scores),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.hook_out"],
        _require_tensor(reference_block.last_output),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.ln2.hook_in"],
        _require_tensor(reference_block.last_resid_mid),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.ln2.hook_out"],
        _require_tensor(reference_block.last_mlp_norm),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.hook_in"],
        _require_tensor(reference_block.last_mlp_norm),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.hook_pattern"],
        _require_tensor(reference_block.last_pattern),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.attn.hook_out"],
        _require_tensor(reference_block.last_attn_output),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.gate.hook_out"],
        _require_tensor(reference_block.last_gate),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.in.hook_out"],
        _require_tensor(reference_block.last_up),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.act.hook_out"],
        _require_tensor(reference_block.last_activation),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.out.hook_in"],
        _require_tensor(reference_block.last_product),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.hook_out"],
        _require_tensor(reference_block.last_mlp_output),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        cache["ln_final.hook_out"],
        _require_tensor(models.reference.model.last_final_norm),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(cache["unembed.hook_out"], reference_logits, rtol=1e-5, atol=1e-6)

    pattern = cache["blocks.0.attn.hook_pattern"]
    assert torch.all(pattern[0, :, 0, -1] > 0)

    changed_future = tokens.clone()
    changed_future[0, -1] = 10
    with torch.inference_mode():
        changed_logits = models.bridge(changed_future)
    assert (bridge_logits[:, 0] - changed_logits[:, 0]).abs().max() > 1e-8


def test_padding_mask_blocks_keys_without_becoming_causal(models: TinyModels) -> None:
    tokens = torch.tensor([[5, 63, 7, 9]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    with torch.inference_mode():
        reference_logits = models.reference(tokens, attention_mask=attention_mask).logits
        bridge_logits, cache = models.bridge.run_with_cache(
            tokens,
            attention_mask=attention_mask,
        )

    torch.testing.assert_close(bridge_logits, reference_logits, rtol=1e-5, atol=1e-6)
    pattern = cache["blocks.0.attn.hook_pattern"]
    torch.testing.assert_close(
        pattern,
        _require_tensor(_reference_block(models.reference).last_pattern),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.count_nonzero(pattern[..., -1]) == 0
    assert torch.all(pattern[0, :, 0, 1:3] > 0)

    changed_padding = tokens.clone()
    changed_padding[0, -1] = 11
    with torch.inference_mode():
        changed_logits = models.bridge(changed_padding, attention_mask=attention_mask)
    torch.testing.assert_close(
        bridge_logits[:, :3],
        changed_logits[:, :3],
        rtol=1e-5,
        atol=1e-6,
    )


def test_run_with_cache_exposes_hooks_without_hf_output_attentions(
    models: TinyModels,
) -> None:
    models.bridge.original_model.output_attentions_requests.clear()
    tokens = torch.tensor([[5, 63, 7, 9]])
    _, cache = models.bridge.run_with_cache(tokens)

    expected_hooks = {
        "embed.hook_out",
        "blocks.0.hook_in",
        "blocks.0.hook_out",
        "blocks.0.ln1.hook_out",
        "blocks.0.attn.hook_in",
        "blocks.0.attn.q.hook_out",
        "blocks.0.attn.k.hook_out",
        "blocks.0.attn.v.hook_out",
        "blocks.0.attn.o.hook_in",
        "blocks.0.attn.o.hook_out",
        "blocks.0.attn.hook_rot_q",
        "blocks.0.attn.hook_rot_k",
        "blocks.0.attn.hook_attn_scores",
        "blocks.0.attn.hook_pattern",
        "blocks.0.attn.hook_out",
        "blocks.0.ln2.hook_out",
        "blocks.0.mlp.hook_in",
        "blocks.0.mlp.gate.hook_out",
        "blocks.0.mlp.in.hook_out",
        "blocks.0.mlp.act.hook_out",
        "blocks.0.mlp.out.hook_in",
        "blocks.0.mlp.hook_out",
        "ln_final.hook_out",
        "unembed.hook_out",
    }
    assert expected_hooks <= set(cache.keys())
    assert models.bridge.original_model.output_attentions_requests == [None]


def test_explicit_hf_output_attentions_request_is_not_silently_suppressed(
    models: TinyModels,
) -> None:
    models.bridge.original_model.output_attentions_requests.clear()
    tokens = torch.tensor([[5, 63, 7, 9]])

    with pytest.raises(ValueError, match="output_attentions is not yet supported"):
        models.bridge(tokens, output_attentions=True)

    assert models.bridge.original_model.output_attentions_requests == [True]


def test_per_head_result_hook_matches_projected_attention(models: TinyModels) -> None:
    tokens = torch.tensor([[5, 63, 7, 9]])
    previous = models.bridge.cfg.use_attn_result
    models.bridge.set_use_attn_result(True)
    try:
        _, cache = models.bridge.run_with_cache(
            tokens,
            names_filter=[
                "blocks.0.attn.hook_result",
                "blocks.0.attn.hook_out",
            ],
        )
    finally:
        models.bridge.set_use_attn_result(previous)

    result = cache["blocks.0.attn.hook_result"]
    projected = cache["blocks.0.attn.hook_out"]
    assert result.shape == (1, 4, 4, 32)
    torch.testing.assert_close(result.sum(dim=2), projected, rtol=1e-5, atol=1e-6)


def test_bridge_wraps_live_direct_block_components(models: TinyModels) -> None:
    bridge_block = models.bridge.blocks[0]
    original_block = bridge_block.original_component
    assert isinstance(original_block, TinyLLaDALlamaBlock)
    assert original_block.q_proj is bridge_block.attn.q
    assert original_block.k_proj is bridge_block.attn.k
    assert original_block.v_proj is bridge_block.attn.v
    assert original_block.attn_out is bridge_block.attn.o
    assert original_block.ff_proj is bridge_block.mlp.gate
    assert original_block.up_proj is getattr(bridge_block.mlp, "in")
    assert original_block.act is bridge_block.mlp.act
    assert original_block.ff_out is bridge_block.mlp.out


def test_complete_reference_state_dict_mapping(models: TinyModels) -> None:
    key_mapping = {
        "model.transformer.wte.weight": "embed.weight",
        "model.transformer.ln_f.weight": "ln_final.weight",
        "model.transformer.ff_out.weight": "unembed.weight",
    }
    for layer in range(2):
        reference_prefix = f"model.transformer.blocks.{layer}"
        bridge_prefix = f"blocks.{layer}"
        key_mapping.update(
            {
                f"{reference_prefix}.attn_norm.weight": f"{bridge_prefix}.ln1.weight",
                f"{reference_prefix}.ff_norm.weight": f"{bridge_prefix}.ln2.weight",
                f"{reference_prefix}.q_proj.weight": f"{bridge_prefix}.attn.q.weight",
                f"{reference_prefix}.k_proj.weight": f"{bridge_prefix}.attn.k.weight",
                f"{reference_prefix}.v_proj.weight": f"{bridge_prefix}.attn.v.weight",
                f"{reference_prefix}.attn_out.weight": f"{bridge_prefix}.attn.o.weight",
                f"{reference_prefix}.ff_proj.weight": f"{bridge_prefix}.mlp.gate.weight",
                f"{reference_prefix}.up_proj.weight": f"{bridge_prefix}.mlp.in.weight",
                f"{reference_prefix}.ff_out.weight": f"{bridge_prefix}.mlp.out.weight",
            }
        )

    reference_state = models.reference.state_dict()
    bridge_state = models.bridge.state_dict()
    assert set(reference_state) == set(key_mapping)
    assert set(bridge_state) == {*key_mapping.values(), "unembed.bias"}
    for reference_key, bridge_key in key_mapping.items():
        torch.testing.assert_close(bridge_state[bridge_key], reference_state[reference_key])
    torch.testing.assert_close(
        bridge_state["unembed.bias"],
        torch.zeros_like(bridge_state["unembed.bias"]),
    )


def test_deepcopy_survives_original_bridge_collection() -> None:
    local = _build_models()
    tokens = torch.tensor([[5, 63, 7, 9]])
    with torch.inference_mode():
        expected_logits = local.bridge(tokens)

    original_block = local.bridge.blocks[0].original_component
    original_block_ref = weakref.ref(original_block)
    clone = copy.deepcopy(local.bridge)
    assert clone.blocks[0].attn.original_component is clone.blocks[0].original_component

    del original_block
    del local
    gc.collect()
    assert original_block_ref() is None

    with torch.inference_mode():
        clone_logits, clone_cache = clone.run_with_cache(tokens)
    torch.testing.assert_close(clone_logits, expected_logits, rtol=1e-5, atol=1e-6)
    assert {
        "blocks.0.attn.hook_in",
        "blocks.0.mlp.hook_in",
        "blocks.0.mlp.hook_out",
    } <= set(clone_cache)


def test_attention_dropout_follows_native_block_mode() -> None:
    local = _build_models(
        TinyLLaDAConfig(attention_dropout=1.0),
        eval_bridge=False,
    )
    assert local.bridge.training is True
    assert local.bridge.blocks[0].original_component.training is False
    tokens = torch.tensor([[5, 63, 7, 9]])

    with torch.inference_mode():
        reference_logits = local.reference(tokens).logits
        bridge_logits = local.bridge(tokens)

    torch.testing.assert_close(bridge_logits, reference_logits, rtol=1e-5, atol=1e-6)


def test_bfloat16_forward_matches_native_sdpa_with_dtype_appropriate_bound() -> None:
    local = _build_models(dtype=torch.bfloat16)
    tokens = torch.tensor([[5, 63, 7, 9]])
    with torch.inference_mode():
        reference_logits = local.reference(tokens).logits
        bridge_logits = local.bridge(tokens)

    max_absolute_drift = (bridge_logits.float() - reference_logits.float()).abs().max().item()
    assert max_absolute_drift < 0.05
    assert torch.equal(bridge_logits.argmax(dim=-1), reference_logits.argmax(dim=-1))


@pytest.mark.parametrize("api_name", ["generate", "generate_stream", "hf_generate"])
def test_autoregressive_generation_apis_fail_clearly(models: TinyModels, api_name: str) -> None:
    tokens = torch.tensor([[5, 63, 7, 9]])
    with pytest.raises(NotImplementedError, match="generation is not supported"):
        if api_name == "generate_stream":
            next(models.bridge.generate_stream(tokens, max_new_tokens=1))
        else:
            getattr(models.bridge, api_name)(tokens, max_new_tokens=1)


@pytest.mark.parametrize("return_type", ["loss", "both"])
def test_causal_loss_modes_fail_clearly(models: TinyModels, return_type: str) -> None:
    tokens = torch.tensor([[5, 63, 7, 9]])
    calls_before = len(models.bridge.original_model.output_attentions_requests)
    with pytest.raises(NotImplementedError, match="causal.*loss|shifted.*loss"):
        models.bridge(tokens, return_type=return_type)
    assert len(models.bridge.original_model.output_attentions_requests) == calls_before
