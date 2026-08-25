"""Stacked weight properties and head labels on encoder-decoder bridges.

The stacking helpers assumed a single top-level ``blocks``; encoder-decoder
adapters register ``encoder_blocks``/``decoder_blocks`` instead, so every
stacked property raised AttributeError and ``all_head_labels`` silently named
only half the heads. Mirrors ``HookedEncoderDecoder``, which stacks
self-attention over ``chain(encoder, decoder)`` and omits cross-attention.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

N_LAYERS = 2
N_HEADS = 4
D_MODEL = 32
D_FF = 64


@pytest.fixture(scope="module")
def t5_bridge() -> TransformerBridge:
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=128,
        d_model=D_MODEL,
        d_kv=D_MODEL // N_HEADS,
        d_ff=D_FF,
        num_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        num_heads=N_HEADS,
    )
    model = T5ForConditionalGeneration(cfg).eval()
    return build_bridge_from_module(
        model,
        "T5ForConditionalGeneration",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()


def test_attention_weights_stack_over_encoder_then_decoder(t5_bridge):
    """W_Q/K/V/O span encoder + decoder layers, not just one stack."""
    total_layers = 2 * N_LAYERS
    d_head = D_MODEL // N_HEADS
    assert t5_bridge.W_Q.shape == (total_layers, N_HEADS, D_MODEL, d_head)
    assert t5_bridge.W_K.shape == (total_layers, N_HEADS, D_MODEL, d_head)
    assert t5_bridge.W_V.shape == (total_layers, N_HEADS, D_MODEL, d_head)
    assert t5_bridge.W_O.shape == (total_layers, N_HEADS, d_head, D_MODEL)


def test_mlp_weights_stack_over_encoder_then_decoder(t5_bridge):
    total_layers = 2 * N_LAYERS
    assert t5_bridge.W_in.shape == (total_layers, D_MODEL, D_FF)
    assert t5_bridge.W_out.shape == (total_layers, D_FF, D_MODEL)


def test_factored_circuits_are_available(t5_bridge):
    """QK/OV build on the stacked weights, so they follow for free."""
    total_layers = 2 * N_LAYERS
    assert t5_bridge.QK.A.shape[0] == total_layers
    assert t5_bridge.OV.A.shape[0] == total_layers


def test_blocks_with_spans_both_stacks(t5_bridge):
    """'attn' means this block's self-attention: encoder attn + decoder self_attn."""
    matching = t5_bridge.blocks_with("attn")
    assert [idx for idx, _ in matching] == list(range(2 * N_LAYERS))


def test_stack_params_for_reports_encoder_and_decoder_layers(t5_bridge):
    indices, stacked = t5_bridge.stack_params_for("attn", "attn.W_Q")
    assert indices == list(range(2 * N_LAYERS))
    assert stacked.shape[0] == 2 * N_LAYERS


def test_all_head_labels_uses_the_encoder_decoder_scheme(t5_bridge):
    """EL/DL labels name every head; a plain L{l}H{h} list named only half."""
    labels = t5_bridge.all_head_labels
    assert len(labels) == 2 * N_LAYERS * N_HEADS
    assert labels[0] == "EL0H0"
    assert labels[-1] == f"DL{N_LAYERS - 1}H{N_HEADS - 1}"
    assert sum(label.startswith("EL") for label in labels) == N_LAYERS * N_HEADS


def test_attn_head_labels_cover_both_stacks(t5_bridge):
    """Derives from composition_layer_indices, which routes through blocks_with."""
    assert len(t5_bridge.attn_head_labels) == 2 * N_LAYERS * N_HEADS


def test_cross_attention_is_excluded_from_stacking(t5_bridge):
    """Decoder blocks also carry cross_attn; HookedEncoderDecoder omits it and so do we."""
    decoder_block = t5_bridge.decoder_blocks[0]
    assert "cross_attn" in decoder_block._modules, "fixture should have cross-attention"
    assert t5_bridge.W_Q.shape[0] == 2 * N_LAYERS


def test_labels_follow_actual_block_counts_when_the_stacks_differ():
    """Asymmetric encoder/decoder depths: labels come from the real block lists.

    cfg.n_layers cannot describe both stacks at once, so deriving labels from it
    would mislabel every decoder head on a lopsided model.
    """
    torch.manual_seed(0)
    encoder_layers, decoder_layers = 3, 1
    cfg = T5Config(
        vocab_size=128,
        d_model=D_MODEL,
        d_kv=D_MODEL // N_HEADS,
        d_ff=D_FF,
        num_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        num_heads=N_HEADS,
    )
    model = T5ForConditionalGeneration(cfg).eval()
    bridge = build_bridge_from_module(
        model,
        "T5ForConditionalGeneration",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()

    labels = bridge.all_head_labels
    assert sum(label.startswith("EL") for label in labels) == encoder_layers * N_HEADS
    assert sum(label.startswith("DL") for label in labels) == decoder_layers * N_HEADS
    assert bridge.W_Q.shape[0] == encoder_layers + decoder_layers
