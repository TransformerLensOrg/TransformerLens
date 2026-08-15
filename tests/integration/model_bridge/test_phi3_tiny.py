"""Tiny-random Phi-3 bridge tests for configs with an explicit ``head_dim``.

Valid Phi-3 configs may set ``head_dim`` independently of
``hidden_size // num_attention_heads`` (the fused QKV width then differs from
``3 * hidden_size``). The adapter previously re-derived ``d_head`` from
``d_model // n_heads`` and crashed splitting the fused matrix during bridge
construction. Uses tiny programmatic configs with real weights — no network
access or weight downloads. Fixture pattern mirrors
tests/integration/model_bridge/test_granite_moe_hybrid_adapter.py.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_config_from_hf
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
)

FP32_TOL = 1e-5

# head_dim is decoupled from hidden_size // num_attention_heads in both:
# MHA derives 32 // 3 = 10 vs explicit 8; GQA derives 32 // 4 = 8 vs explicit 6.
GEOMETRIES = {
    "mha": dict(num_attention_heads=3, num_key_value_heads=3, head_dim=8),
    "gqa": dict(num_attention_heads=4, num_key_value_heads=2, head_dim=6),
}


class _MockTokenizer:
    """Stand-in to satisfy TransformerBridge(tokenizer=...)."""

    pass


@pytest.fixture(scope="module", params=sorted(GEOMETRIES), ids=sorted(GEOMETRIES))
def geometry(request):
    """One attention geometry (MHA or GQA), each with a decoupled head_dim."""
    return GEOMETRIES[request.param]


@pytest.fixture(scope="module")
def hf_model(geometry):
    hf_config = AutoConfig.for_model(
        "phi3",
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **geometry,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager")
    return model.eval()


@pytest.fixture(scope="module")
def tokens():
    # Ids stay below the tiny vocab_size=100.
    return torch.tensor([[1, 5, 7, 42, 9]])


@pytest.fixture(scope="module")
def hf_logits(hf_model, tokens):
    """Reference logits, captured before the bridge wraps (and mutates) the HF module.

    This keeps them an independent ground truth — helpers.assert_bridge_matches_hf
    compares against the already-wrapped model, so it can't provide that.
    """
    with torch.no_grad():
        return hf_model(tokens).logits


@pytest.fixture(scope="module")
def bridge(hf_model, hf_logits):
    """TransformerBridge over the tiny HF model.

    The config goes through build_bridge_config_from_hf — the same translation
    path boot_transformers uses, including the explicit-head_dim mapping under
    test. Depends on hf_logits so the reference capture happens before wrapping.
    """
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, "Phi3ForCausalLM", "phi3-tiny", torch.float32
    )
    adapter = Phi3ArchitectureAdapter(bridge_config)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=_MockTokenizer())


def test_mapped_d_head_honours_explicit_head_dim(bridge, geometry) -> None:
    assert bridge.cfg.d_head == geometry["head_dim"]


def test_split_qkv_weights_match_fused_slices(bridge, geometry) -> None:
    """Q/K/V weights must be bitwise slices of the original fused qkv_proj."""
    head_dim = geometry["head_dim"]
    q_size = bridge.cfg.n_heads * head_dim
    kv_size = (bridge.cfg.n_key_value_heads or bridge.cfg.n_heads) * head_dim
    for block in bridge.blocks:
        fused = block.attn.qkv.weight
        assert fused.shape[0] == q_size + 2 * kv_size
        assert torch.equal(block.attn.q.weight, fused[:q_size])
        assert torch.equal(block.attn.k.weight, fused[q_size : q_size + kv_size])
        assert torch.equal(block.attn.v.weight, fused[q_size + kv_size :])


def test_forward_matches_hf_eager(bridge, hf_logits, tokens) -> None:
    with torch.no_grad():
        bridge_logits = bridge(tokens)
    max_diff = (bridge_logits - hf_logits).abs().max().item()
    assert max_diff < FP32_TOL, (
        f"bridge vs HF eager drift={max_diff:.2e} exceeds {FP32_TOL:.0e} "
        f"for explicit-head_dim Phi-3"
    )


def test_run_with_cache_head_shapes(bridge, tokens, geometry) -> None:
    """Cached per-head activations must use the explicit head_dim."""
    head_dim = geometry["head_dim"]
    _, cache = bridge.run_with_cache(tokens)
    n_kv = bridge.cfg.n_key_value_heads or bridge.cfg.n_heads
    batch, seq = tokens.shape
    z = cache["blocks.0.attn.hook_z"]
    assert z.shape == (batch, seq, bridge.cfg.n_heads, head_dim)
    k = cache["blocks.0.attn.k.hook_out"]
    assert k.shape[-2:] == (n_kv, head_dim)
