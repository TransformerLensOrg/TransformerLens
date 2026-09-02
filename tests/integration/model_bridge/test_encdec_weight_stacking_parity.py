"""Stacked enc-dec weights on the bridge match the raw HF T5 weights.

Deletion evidence for HookedEncoderDecoder's stacked-weight properties.
HookedEncoderDecoder did no weight processing — its stacks were pure reshapes
of the HF tensors — so the raw HF model is the same oracle without the legacy
class: expected tensors are derived here directly from t5-small's weights with
the documented TL orientations, over chain(encoder, decoder).
"""

from __future__ import annotations

import einops
import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "google-t5/t5-small"


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def hf_blocks(bridge):
    """chain(encoder, decoder) HF blocks — the order the bridge stacks over."""
    hf = bridge.original_model
    return list(hf.encoder.block) + list(hf.decoder.block)


def _self_attn(block):
    return block.layer[0].SelfAttention


def _mlp(block):
    # T5 FF layer is the last entry (index 1 on encoder blocks, 2 on decoder).
    return block.layer[-1].DenseReluDense


def _expected(name, blocks, n_heads):
    per_block = []
    for b in blocks:
        if name in ("W_Q", "W_K", "W_V"):
            hf_w = getattr(_self_attn(b), {"W_Q": "q", "W_K": "k", "W_V": "v"}[name]).weight
            per_block.append(einops.rearrange(hf_w, "(h d) m -> h m d", h=n_heads))
        elif name == "W_O":
            per_block.append(
                einops.rearrange(_self_attn(b).o.weight, "m (h d) -> h d m", h=n_heads)
            )
        elif name == "W_in":
            per_block.append(_mlp(b).wi.weight.T)
        elif name == "W_out":
            per_block.append(_mlp(b).wo.weight.T)
    return torch.stack(per_block, dim=0)


@pytest.mark.parametrize("name", ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"])
def test_stacked_weights_match_raw_hf(name, bridge, hf_blocks):
    expected = _expected(name, hf_blocks, bridge.cfg.n_heads)
    actual = getattr(bridge, name)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_head_labels_cover_both_stacks(bridge):
    """EL{l}H{h} then DL{l}H{h}, sized by the real block lists."""
    n_enc = len(bridge.original_model.encoder.block)
    n_dec = len(bridge.original_model.decoder.block)
    heads = range(bridge.cfg.n_heads)
    expected = [f"EL{l}H{h}" for l in range(n_enc) for h in heads] + [
        f"DL{l}H{h}" for l in range(n_dec) for h in heads
    ]
    assert bridge.all_head_labels == expected
