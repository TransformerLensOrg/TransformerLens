"""Convenience-property tests for ``HookedEncoder``.

Closes the last open ask in #277 — verify each ``W_*`` / ``b_*`` / circuit
property has the right shape AND aliases the right underlying parameter, so
property-level mech-interp work doesn't silently read the wrong tensor.

Uses a randomly-initialized small encoder (no HF download) so the tests run
fast and deterministically.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens import FactoredMatrix, HookedEncoder, HookedTransformerConfig

D_MODEL = 12
D_HEAD = 4
N_HEADS = D_MODEL // D_HEAD
D_MLP = 4 * D_MODEL
N_CTX = 5
N_LAYERS = 3
D_VOCAB = 22


@pytest.fixture
def model() -> HookedEncoder:
    cfg = HookedTransformerConfig(
        d_head=D_HEAD,
        d_model=D_MODEL,
        n_ctx=N_CTX,
        n_layers=N_LAYERS,
        act_fn="gelu",
        d_vocab=D_VOCAB,
    )
    encoder = HookedEncoder(cfg)
    # HookedEncoder uses torch.empty() for params and does no init pass; the
    # uninitialized memory contains NaNs which break torch.equal comparisons.
    torch.manual_seed(0)
    for p in encoder.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return encoder


# ---------------------------------------------------------------------------
# Embed / unembed
# ---------------------------------------------------------------------------


def test_W_U(model: HookedEncoder):
    assert model.W_U.shape == (D_MODEL, D_VOCAB)
    assert model.W_U is model.unembed.W_U


def test_b_U(model: HookedEncoder):
    assert model.b_U.shape == (D_VOCAB,)
    assert model.b_U is model.unembed.b_U


def test_W_E(model: HookedEncoder):
    assert model.W_E.shape == (D_VOCAB, D_MODEL)
    assert model.W_E is model.embed.embed.W_E


def test_W_pos(model: HookedEncoder):
    assert model.W_pos.shape == (N_CTX, D_MODEL)
    assert model.W_pos is model.embed.pos_embed.W_pos


@pytest.mark.xfail(
    reason=(
        "HookedEncoder.W_E_pos return annotation 'd_vocab+n_ctx d_model' references "
        "unbound dimension names (no input args supply them), so the jaxtyping import-hook "
        "can't resolve the sum at runtime. Same annotation exists on HookedTransformer.W_E_pos; "
        "fixing it is a separate API-touch."
    ),
    strict=True,
)
def test_W_E_pos(model: HookedEncoder):
    assert model.W_E_pos.shape == (D_VOCAB + N_CTX, D_MODEL)
    # Concatenation, so identity doesn't apply — verify the slices match.
    assert torch.equal(model.W_E_pos[:D_VOCAB], model.W_E)
    assert torch.equal(model.W_E_pos[D_VOCAB:], model.W_pos)


# ---------------------------------------------------------------------------
# Per-layer attention weights/biases — stacked across blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attr", ["W_Q", "W_K", "W_V"])
def test_attn_qkv_weight(model: HookedEncoder, attr: str):
    stacked = getattr(model, attr)
    assert stacked.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(stacked[layer_idx], getattr(block.attn, attr))


def test_W_O(model: HookedEncoder):
    assert model.W_O.shape == (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(model.W_O[layer_idx], block.attn.W_O)


@pytest.mark.parametrize("attr", ["b_Q", "b_K", "b_V"])
def test_attn_qkv_bias(model: HookedEncoder, attr: str):
    stacked = getattr(model, attr)
    assert stacked.shape == (N_LAYERS, N_HEADS, D_HEAD)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(stacked[layer_idx], getattr(block.attn, attr))


def test_b_O(model: HookedEncoder):
    assert model.b_O.shape == (N_LAYERS, D_MODEL)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(model.b_O[layer_idx], block.attn.b_O)


# ---------------------------------------------------------------------------
# Per-layer MLP weights/biases — stacked across blocks
# ---------------------------------------------------------------------------


def test_W_in(model: HookedEncoder):
    assert model.W_in.shape == (N_LAYERS, D_MODEL, D_MLP)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(model.W_in[layer_idx], block.mlp.W_in)


def test_W_out(model: HookedEncoder):
    assert model.W_out.shape == (N_LAYERS, D_MLP, D_MODEL)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(model.W_out[layer_idx], block.mlp.W_out)


def test_b_in(model: HookedEncoder):
    assert model.b_in.shape == (N_LAYERS, D_MLP)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(model.b_in[layer_idx], block.mlp.b_in)


def test_b_out(model: HookedEncoder):
    assert model.b_out.shape == (N_LAYERS, D_MODEL)
    for layer_idx, block in enumerate(model.blocks):
        assert torch.equal(model.b_out[layer_idx], block.mlp.b_out)


# ---------------------------------------------------------------------------
# Factored circuits
# ---------------------------------------------------------------------------


def test_QK_circuit(model: HookedEncoder):
    qk = model.QK
    assert isinstance(qk, FactoredMatrix)
    # Left factor is W_Q [..., d_model, d_head]; right factor is W_K transposed
    # to [..., d_head, d_model]. Their product would be [..., d_model, d_model].
    assert qk.A.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    assert qk.B.shape == (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)
    assert torch.equal(qk.A, model.W_Q)
    assert torch.equal(qk.B, model.W_K.transpose(-2, -1))


def test_OV_circuit(model: HookedEncoder):
    ov = model.OV
    assert isinstance(ov, FactoredMatrix)
    assert ov.A.shape == (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
    assert ov.B.shape == (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)
    assert torch.equal(ov.A, model.W_V)
    assert torch.equal(ov.B, model.W_O)
