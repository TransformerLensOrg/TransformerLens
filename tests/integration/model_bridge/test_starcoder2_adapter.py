"""Integration tests for the Starcoder2 architecture adapter.

Builds a tiny ``Starcoder2ForCausalLM`` from config (no download) and checks that
the TransformerBridge reproduces its structure and — most importantly — its
forward-pass logits. Starcoder2 exercises a LayerNorm + non-gated-MLP + biased
+ GQA decoder, so a logit match is strong evidence the adapter's weight
conversions and component mapping are correct.
"""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer, Starcoder2Config, Starcoder2ForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge

D_MODEL = 64
N_HEADS = 8
N_KV_HEADS = 2
N_LAYERS = 2


def _make_hf_model() -> Starcoder2ForCausalLM:
    cfg = Starcoder2Config(
        hidden_size=D_MODEL,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        max_position_embeddings=128,
        vocab_size=1000,
        use_bias=True,
    )
    torch.manual_seed(0)
    model = Starcoder2ForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_and_bridge():
    """Return a matched (hf_model, bridge) pair loaded from the same weights."""
    hf_model = _make_hf_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_model.save_pretrained(tmpdir)
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(tmpdir)
        bridge = TransformerBridge.boot_transformers(tmpdir, device="cpu")
    bridge.eval()
    return hf_model, bridge


def _tokens() -> torch.Tensor:
    return torch.tensor([[5, 12, 7, 99, 3, 42, 8, 1]])


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestStarcoder2BridgeStructure:
    def test_block_count(self, hf_and_bridge) -> None:
        _, bridge = hf_and_bridge
        assert len(bridge.blocks) == N_LAYERS

    def test_attention_is_separate_qkv(self, hf_and_bridge) -> None:
        _, bridge = hf_and_bridge
        attn = bridge.blocks[0].attn
        for proj in ("q", "k", "v", "o"):
            assert hasattr(attn, proj)

    def test_mlp_is_non_gated(self, hf_and_bridge) -> None:
        _, bridge = hf_and_bridge
        mlp = bridge.blocks[0].mlp
        assert hasattr(mlp, "in")
        assert hasattr(mlp, "out")
        assert not hasattr(mlp, "gate")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestStarcoder2ForwardPass:
    def test_forward_returns_correct_shape(self, hf_and_bridge) -> None:
        _, bridge = hf_and_bridge
        logits = bridge(_tokens())
        assert logits.shape == (1, 8, 1000)

    def test_forward_matches_hf(self, hf_and_bridge) -> None:
        """The decisive test: bridge logits must equal HuggingFace's up to float noise."""
        hf_model, bridge = hf_and_bridge
        tokens = _tokens()
        with torch.no_grad():
            hf_logits = hf_model(tokens).logits
            bridge_logits = bridge(tokens)
        assert torch.allclose(hf_logits, bridge_logits, atol=1e-4, rtol=1e-4)

    def test_forward_produces_no_nans(self, hf_and_bridge) -> None:
        _, bridge = hf_and_bridge
        logits = bridge(_tokens())
        assert not torch.isnan(logits).any()
