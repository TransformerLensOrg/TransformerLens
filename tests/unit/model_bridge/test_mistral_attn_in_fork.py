"""Mistral adapter uses a fork-capable attention bridge (per-receiver ``hook_attn_in``).

Mistral has separate q/k/v/o projections with RoPE + GQA, identical in structure to Qwen2, but
its adapter previously used the plain ``AttentionBridge`` (which delegates q/k/v to HF and exposes
no fork point), so ``set_use_attn_in`` raised and ``blocks.{i}.hook_attn_in`` did not exist. It now
uses ``PositionEmbeddingsAttentionBridge`` like Qwen2, enabling the input fork the circuit-analysis
tooling relies on. These tests boot the tiny random Mistral so they stay cheap.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

_MODEL = "hf-internal-testing/tiny-random-MistralForCausalLM"


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(_MODEL, device="cpu")


@pytest.mark.slow
def test_mistral_uses_fork_capable_attention(bridge: TransformerBridge) -> None:
    assert isinstance(bridge.blocks[0].attn, PositionEmbeddingsAttentionBridge)


@pytest.mark.slow
def test_mistral_hook_attn_in_shape_and_intervenes(bridge: TransformerBridge) -> None:
    toks = torch.arange(1, 7).unsqueeze(0)
    baseline = bridge(toks)

    bridge.set_use_attn_in(True)  # previously raised on the plain AttentionBridge

    captured: dict = {}

    def _grab(tensor: torch.Tensor, hook: object) -> torch.Tensor:
        captured["shape"] = tuple(tensor.shape)
        return tensor

    bridge.run_with_hooks(toks, fwd_hooks=[("blocks.0.hook_attn_in", _grab)])
    # [batch, pos, n_heads, d_model]
    assert captured["shape"] == (1, 6, bridge.cfg.n_heads, bridge.cfg.d_model)

    def _zero_head0(tensor: torch.Tensor, hook: object) -> torch.Tensor:
        tensor = tensor.clone()
        tensor[:, :, 0, :] = 0.0
        return tensor

    ablated = bridge.run_with_hooks(toks, fwd_hooks=[("blocks.0.hook_attn_in", _zero_head0)])
    # zeroing a single head's forked input must actually change the output
    assert (ablated - baseline).abs().max().item() > 1e-6
