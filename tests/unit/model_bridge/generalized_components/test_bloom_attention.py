"""Unit tests for BloomAttentionBridge's reconstructed attention.

Wraps a programmatically built HF BloomAttention — no Hub download.
"""

import torch
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BloomAttentionBridge,
    LinearBridge,
)
from transformer_lens.model_bridge.supported_architectures.bloom import (
    BloomArchitectureAdapter,
)

D_MODEL = 32
N_HEADS = 4
BATCH = 2
SEQ = 6


def _build_bridge() -> BloomAttentionBridge:
    """Wire a live BloomAttention into BloomAttentionBridge with the real QKV split."""
    cfg = TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=1,
        n_ctx=32,
        n_heads=N_HEADS,
        d_vocab=99,
        architecture="BloomForCausalLM",
    )
    adapter = BloomArchitectureAdapter(cfg)
    hf_attn = BloomAttention(
        BloomConfig(hidden_size=D_MODEL, n_head=N_HEADS, n_layer=1, vocab_size=99), layer_idx=0
    )
    hf_attn.eval()

    bridge = BloomAttentionBridge(
        name="self_attention",
        config=adapter.cfg,
        split_qkv_matrix=adapter.split_qkv_matrix,
        submodules={
            "qkv": LinearBridge(name="query_key_value"),
            "o": LinearBridge(name="dense"),
        },
    )
    bridge.set_original_component(hf_attn)
    # The full bridge wires submodules during model setup; do it by hand here.
    bridge.o.set_original_component(hf_attn.dense)
    return bridge


def _captured_scores(compatibility_mode: bool) -> torch.Tensor:
    """Run one forward with HF's additive finfo.min causal mask and grab hook_attn_scores."""
    bridge = _build_bridge()
    bridge.compatibility_mode = compatibility_mode

    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    alibi = torch.zeros(BATCH * N_HEADS, 1, SEQ)
    causal = torch.triu(torch.full((SEQ, SEQ), torch.finfo(torch.float32).min), diagonal=1)
    attention_mask = causal[None, None].expand(BATCH, 1, -1, -1)

    captured: dict[str, torch.Tensor] = {}
    bridge.hook_attn_scores.add_hook(
        lambda tensor, hook: captured.setdefault("scores", tensor.detach().clone())
    )
    with torch.no_grad():
        bridge(hidden, torch.zeros_like(hidden), alibi=alibi, attention_mask=attention_mask)
    return captured["scores"]


def _masked_positions(scores: torch.Tensor) -> torch.Tensor:
    return torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1).expand_as(scores)


class TestCompatibilityMaskSentinel:
    """Compatibility mode must report masked scores as -inf, not HF's finfo.min."""

    def test_masked_scores_are_negative_infinity(self) -> None:
        """`torch.isinf(cache[...hook_attn_scores])` is the documented way to find
        masked positions; a finfo.min sentinel makes it silently return all-False."""
        scores = _captured_scores(compatibility_mode=True)
        masked = _masked_positions(scores)
        assert masked.any()
        assert torch.isneginf(scores[masked]).all()
        assert torch.isfinite(scores[~masked]).all()

    def test_non_compatibility_mode_keeps_the_hf_sentinel(self) -> None:
        """Outside compatibility mode the bridge must leave HF's finfo.min alone."""
        scores = _captured_scores(compatibility_mode=False)
        assert torch.isfinite(scores[_masked_positions(scores)]).all()
