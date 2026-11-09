"""XGLM adds a sinusoidal position table to the scaled token embedding before block 0.

The bridge must expose it as ``hook_pos_embed`` so the residual-decomposition APIs
account for it.
"""
import pytest
import torch

from tests.integration.model_bridge.helpers import make_tiny_pair

VOCAB = 128
TOKENS = torch.tensor([[2, 5, 9, 17, 33, 65, 3, 4, 6, 7], [2, 11, 13, 19, 23, 29, 31, 37, 41, 43]])


@pytest.fixture(scope="module")
def pair():
    from transformers import XGLMConfig

    cfg = XGLMConfig(
        vocab_size=VOCAB,
        d_model=64,
        ffn_dim=128,
        num_layers=3,
        attention_heads=4,
        max_position_embeddings=64,
    )
    return make_tiny_pair(cfg, "XGLMForCausalLM")


@pytest.fixture(scope="module")
def cache(pair):
    bridge, _ = pair
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    return cache


def test_forward_matches_hf(pair):
    bridge, ref = pair
    with torch.no_grad():
        torch.testing.assert_close(bridge(TOKENS), ref(TOKENS).logits, atol=1e-5, rtol=0)


def test_embed_plus_pos_embed_is_first_residual(cache):
    torch.testing.assert_close(
        cache["hook_embed"] + cache["hook_pos_embed"],
        cache["blocks.0.hook_resid_pre"],
        atol=1e-6,
        rtol=0,
    )


def test_pos_embed_is_the_offset_sinusoidal_table(pair, cache):
    bridge, ref = pair
    table = ref.model.embed_positions.weights
    offset = ref.model.embed_positions.offset
    expected = table[offset : offset + TOKENS.shape[1]].expand(TOKENS.shape[0], -1, -1)
    torch.testing.assert_close(cache["hook_pos_embed"], expected)
    torch.testing.assert_close(bridge.pos_embed.W_pos, table)


@pytest.mark.parametrize("layer", [0, 1, 2, 3])
def test_decompose_resid_sums_to_resid(pair, cache, layer):
    bridge, _ = pair
    n = bridge.cfg.n_layers
    target = cache[
        f"blocks.{n - 1}.hook_resid_post" if layer == n else f"blocks.{layer}.hook_resid_pre"
    ]
    torch.testing.assert_close(cache.decompose_resid(layer=layer).sum(0), target, atol=1e-5, rtol=0)


@pytest.mark.parametrize("expand_neurons", [False, True])
def test_full_resid_decomposition_sums_to_final_resid(pair, cache, expand_neurons):
    bridge, _ = pair
    stack, labels = cache.get_full_resid_decomposition(
        expand_neurons=expand_neurons, return_labels=True
    )
    assert "pos_embed" in labels
    final = cache[f"blocks.{bridge.cfg.n_layers - 1}.hook_resid_post"]
    torch.testing.assert_close(stack.sum(0), final, atol=1e-5, rtol=0)


def test_greedy_generate_matches_hf(pair):
    """Cached decoding passes past_key_values_length to the position table."""
    bridge, ref = pair
    prompt = TOKENS[:1, :4]
    with torch.no_grad():
        expected = ref.generate(prompt, max_new_tokens=6, do_sample=False)
        out = bridge.generate(
            prompt, max_new_tokens=6, do_sample=False, return_type="tokens", verbose=False
        )
    assert torch.equal(out, expected)
