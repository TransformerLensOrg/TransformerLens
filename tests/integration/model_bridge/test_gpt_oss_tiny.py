"""Tiny-random GPT-OSS bridge-vs-HF parity (issue #1618).

GPT-OSS's eager attention folds a learned per-head sink logit into the softmax;
the bridge's reconstructed attention must carry it or every layer's pattern is
rescaled. A tiny unquantized ``GptOssForCausalLM`` exercises the same code path
CPU-fast — sinks exist independent of MXFP4 quantization.
"""

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GptOssConfig,
    GptOssForCausalLM,
)

from transformer_lens.model_bridge.bridge import TransformerBridge

# Ids stay below the tiny vocab_size=256.
TOKENS = torch.tensor([[5, 17, 29, 3, 11, 42, 7, 23]])
FP32_TOL = 1e-4
N_LAYERS = 2


def _tiny_config() -> GptOssConfig:
    return GptOssConfig(
        hidden_size=64,
        num_hidden_layers=N_LAYERS,  # default layer_types: one sliding, one full
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
        sliding_window=4,  # < seq len so the sliding-window mask actually bites
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
    )


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    model = GptOssForCausalLM(_tiny_config())
    path = tmp_path_factory.mktemp("tiny_gpt_oss")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def hf_eager(model_dir):
    """Reference model loaded independently of the bridge's wrapped instance."""
    return AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def bridge(model_dir):
    return TransformerBridge.boot_transformers(model_dir, device="cpu", dtype=torch.float32)


def test_random_init_gives_nonzero_sinks(hf_eager):
    """Guard: zero sinks would make the parity tests vacuously pass without them."""
    sinks = hf_eager.model.layers[0].self_attn.sinks
    assert sinks.abs().max() > 0


def test_logits_match_hf_eager(bridge, hf_eager):
    with torch.inference_mode():
        bridge_logits = bridge(TOKENS)
        hf_logits = hf_eager(TOKENS).logits
    max_diff = (bridge_logits - hf_logits).abs().max().item()
    assert max_diff < FP32_TOL, (
        f"bridge vs HF eager logit drift={max_diff:.2e} exceeds {FP32_TOL:.0e} — "
        "reconstructed attention diverges (attention sinks? see #1618)"
    )


def test_resid_post_matches_hf_layer_outputs(bridge, hf_eager):
    """Per-layer parity catches compensating errors that wash out at the logits.

    Compares against forward hooks on the HF decoder layers — NOT
    ``output_hidden_states``, whose final entry is post-``model.norm`` and so
    never matches the last block's resid_post.
    """
    hf_layer_out: dict[int, torch.Tensor] = {}

    def _make_hook(idx):
        def _hook(_module, _inputs, output):
            hf_layer_out[idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return _hook

    handles = [
        layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(hf_eager.model.layers)
    ]
    try:
        with torch.inference_mode():
            hf_eager(TOKENS)
    finally:
        for handle in handles:
            handle.remove()

    with torch.inference_mode():
        _, cache = bridge.run_with_cache(
            TOKENS, names_filter=lambda n: n.endswith("hook_resid_post")
        )
    for layer in range(N_LAYERS):
        diff = (hf_layer_out[layer] - cache[f"blocks.{layer}.hook_resid_post"]).abs().max()
        assert diff < FP32_TOL, f"layer {layer} resid_post drift={diff:.2e}"


def test_pattern_rows_leave_mass_for_the_sink(bridge):
    """hook_pattern rows must sum to < 1: the dropped sink column keeps its share."""
    with torch.inference_mode():
        _, cache = bridge.run_with_cache(
            TOKENS, names_filter=lambda n: n.endswith("attn.hook_pattern")
        )
    for layer in range(N_LAYERS):
        row_sums = cache[f"blocks.{layer}.attn.hook_pattern"].sum(dim=-1)
        assert (row_sums < 1.0).all(), f"layer {layer}: a pattern row ignored the sink's share"
