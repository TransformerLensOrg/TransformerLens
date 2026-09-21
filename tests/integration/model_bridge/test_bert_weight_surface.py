"""BERT weight-accessor surface on TransformerBridge.

The bridge successor to tests/integration/test_hooked_encoder_properties.py
(#277): every ``W_*`` / ``b_*`` / circuit accessor must have the right shape
AND carry the right underlying HF parameter in TL orientation, so
property-level mech-interp work doesn't silently read the wrong tensor. That
legacy suite certifies the surface only on HookedEncoder and is deleted with
the class at 4.0; this file is the coverage that survives.

Uses a tiny randomly-initialized BertForMaskedLM (no download).
"""

from __future__ import annotations

import copy

import einops
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

D_MODEL = 12
D_HEAD = 4
N_HEADS = D_MODEL // D_HEAD
D_MLP = 4 * D_MODEL
N_CTX = 5
N_LAYERS = 3
D_VOCAB = 22


@pytest.fixture(scope="module")
def hf_model() -> BertForMaskedLM:
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=D_VOCAB,
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=D_MLP,
        max_position_embeddings=N_CTX,
    )
    return BertForMaskedLM(cfg).eval()


@pytest.fixture(scope="module")
def model(hf_model):
    return build_bridge_from_module(
        hf_model,
        "BertForMaskedLM",
        hf_config=copy.deepcopy(hf_model.config),
        tokenizer=None,
        device="cpu",
    )


def _layer(hf_model, i):
    return hf_model.bert.encoder.layer[i]


# ---------------------------------------------------------------------------
# Embed / unembed
# ---------------------------------------------------------------------------


def test_W_E(model, hf_model):
    assert model.W_E.shape == (D_VOCAB, D_MODEL)
    torch.testing.assert_close(
        model.W_E, hf_model.bert.embeddings.word_embeddings.weight, atol=0.0, rtol=0.0
    )


def test_W_pos(model, hf_model):
    assert model.W_pos.shape == (N_CTX, D_MODEL)
    torch.testing.assert_close(
        model.W_pos, hf_model.bert.embeddings.position_embeddings.weight, atol=0.0, rtol=0.0
    )


def test_W_E_pos(model):
    assert model.W_E_pos.shape == (D_VOCAB + N_CTX, D_MODEL)
    assert torch.equal(model.W_E_pos[:D_VOCAB], model.W_E)
    assert torch.equal(model.W_E_pos[D_VOCAB:], model.W_pos)


def test_W_U(model):
    assert model.W_U.shape == (D_MODEL, D_VOCAB)


def test_b_U(model):
    assert model.b_U.shape == (D_VOCAB,)


# ---------------------------------------------------------------------------
# Attention stacks — value-checked against the HF parameters in TL orientation
# ---------------------------------------------------------------------------

_QKV_HF_ATTR = {"W_Q": "query", "W_K": "key", "W_V": "value"}


@pytest.mark.parametrize("attr", ["W_Q", "W_K", "W_V"])
def test_attn_qkv_weight(model, hf_model, attr):
    stacked = getattr(model, attr)
    assert stacked.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    for i in range(N_LAYERS):
        hf_w = getattr(_layer(hf_model, i).attention.self, _QKV_HF_ATTR[attr]).weight
        expected = einops.rearrange(hf_w, "(h d) m -> h m d", h=N_HEADS)
        torch.testing.assert_close(stacked[i], expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("attr", ["b_Q", "b_K", "b_V"])
def test_attn_qkv_bias(model, hf_model, attr):
    stacked = getattr(model, attr)
    assert stacked.shape == (N_LAYERS, N_HEADS, D_HEAD)
    for i in range(N_LAYERS):
        hf_b = getattr(_layer(hf_model, i).attention.self, _QKV_HF_ATTR["W_" + attr[-1]]).bias
        expected = einops.rearrange(hf_b, "(h d) -> h d", h=N_HEADS)
        torch.testing.assert_close(stacked[i], expected, atol=0.0, rtol=0.0)


def test_W_O(model, hf_model):
    assert model.W_O.shape == (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)
    for i in range(N_LAYERS):
        hf_w = _layer(hf_model, i).attention.output.dense.weight
        expected = einops.rearrange(hf_w, "m (h d) -> h d m", h=N_HEADS)
        torch.testing.assert_close(model.W_O[i], expected, atol=0.0, rtol=0.0)


def test_b_O(model, hf_model):
    assert model.b_O.shape == (N_LAYERS, D_MODEL)
    for i in range(N_LAYERS):
        torch.testing.assert_close(
            model.b_O[i], _layer(hf_model, i).attention.output.dense.bias, atol=0.0, rtol=0.0
        )


# ---------------------------------------------------------------------------
# MLP stacks
# ---------------------------------------------------------------------------


def test_W_in(model, hf_model):
    assert model.W_in.shape == (N_LAYERS, D_MODEL, D_MLP)
    for i in range(N_LAYERS):
        expected = _layer(hf_model, i).intermediate.dense.weight.T
        torch.testing.assert_close(model.W_in[i], expected, atol=0.0, rtol=0.0)


def test_W_out(model, hf_model):
    assert model.W_out.shape == (N_LAYERS, D_MLP, D_MODEL)
    for i in range(N_LAYERS):
        expected = _layer(hf_model, i).output.dense.weight.T
        torch.testing.assert_close(model.W_out[i], expected, atol=0.0, rtol=0.0)


def test_b_in(model, hf_model):
    assert model.b_in.shape == (N_LAYERS, D_MLP)
    for i in range(N_LAYERS):
        torch.testing.assert_close(
            model.b_in[i], _layer(hf_model, i).intermediate.dense.bias, atol=0.0, rtol=0.0
        )


def test_b_out(model, hf_model):
    assert model.b_out.shape == (N_LAYERS, D_MODEL)
    for i in range(N_LAYERS):
        torch.testing.assert_close(
            model.b_out[i], _layer(hf_model, i).output.dense.bias, atol=0.0, rtol=0.0
        )


# ---------------------------------------------------------------------------
# Factored circuits
# ---------------------------------------------------------------------------


def test_QK_circuit(model):
    qk = model.QK
    assert isinstance(qk, FactoredMatrix)
    assert qk.A.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    assert qk.B.shape == (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)
    assert torch.equal(qk.A, model.W_Q)
    assert torch.equal(qk.B, model.W_K.transpose(-2, -1))


def test_OV_circuit(model):
    ov = model.OV
    assert isinstance(ov, FactoredMatrix)
    assert ov.A.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    assert ov.B.shape == (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)
    assert torch.equal(ov.A, model.W_V)
    assert torch.equal(ov.B, model.W_O)
