"""Per-head attention results on JetMoE's mixture of attention heads.

Each token routes to ``top_k`` attention experts; expert slot ``k`` owns heads
``k * n_kv_heads .. (k + 1) * n_kv_heads - 1``, and its output projection and gate
depend on the token's routing. The bridge exposes ``hook_z`` at the input of
``experts.reduce`` and computes per-head results from it and the router logits.
"""
import copy

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, JetMoeConfig

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

TOKENS = torch.tensor([[3, 17, 42, 5, 99, 64, 7], [11, 2, 120, 33, 8, 77, 50]])
ATOL = 1e-5


def _pair():
    cfg = JetMoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_key_value_heads=2,
        kv_channels=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()
    with torch.no_grad():
        for name, param in hf.named_parameters():
            if name.endswith("experts.bias"):  # the MoA output bias, b_O
                param.normal_(0.0, 0.5)
    ref = copy.deepcopy(hf)
    bridge = build_bridge_from_module(hf, "JetMoeForCausalLM", hf_config=copy.deepcopy(hf.config))
    return bridge, ref


@pytest.fixture(scope="module")
def pair():
    return _pair()


@pytest.fixture(scope="module")
def cache(pair):
    bridge, _ = pair
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    return cache


def test_forward_matches_hf(pair):
    bridge, ref = pair
    with torch.no_grad():
        torch.testing.assert_close(bridge(TOKENS), ref(TOKENS).logits, atol=ATOL, rtol=0)


def test_hook_z_is_per_head(pair, cache):
    bridge, _ = pair
    cfg = bridge.cfg
    assert cache["blocks.0.attn.hook_z"].shape == (*TOKENS.shape, cfg.n_heads, cfg.d_head)


def test_hook_z_patching_changes_the_output(pair):
    bridge, _ = pair

    def ablate_head(z, hook):
        z = z.clone()
        z[:, :, 1] = 0.0
        return z

    with torch.no_grad():
        clean = bridge(TOKENS)
        ablated = bridge.run_with_hooks(TOKENS, fwd_hooks=[("blocks.1.attn.hook_z", ablate_head)])
        identity = bridge.run_with_hooks(
            TOKENS, fwd_hooks=[("blocks.1.attn.hook_z", lambda z, hook: z)]
        )
    torch.testing.assert_close(identity, clean, atol=0, rtol=0)
    assert not torch.allclose(ablated, clean, atol=1e-4)


def test_heads_plus_bias_sum_to_attn_out(pair, cache):
    bridge, _ = pair
    n_heads = bridge.cfg.n_heads
    for layer in range(bridge.cfg.n_layers):
        heads = cache.stack_head_results(layer=layer + 1)[layer * n_heads :]
        torch.testing.assert_close(
            heads.sum(0) + bridge.blocks[layer].attn.b_O,
            cache[f"blocks.{layer}.hook_attn_out"],
            atol=ATOL,
            rtol=0,
        )


def test_heads_plus_remainder_sum_to_resid(pair, cache):
    bridge, _ = pair
    n = bridge.cfg.n_layers
    stack = cache.stack_head_results(layer=n, incl_remainder=True)
    torch.testing.assert_close(
        stack.sum(0), cache[f"blocks.{n - 1}.hook_resid_post"], atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("layer", [1, 3])
def test_full_resid_decomposition_sums_to_resid(pair, cache, layer):
    bridge, _ = pair
    n = bridge.cfg.n_layers
    target = cache[
        f"blocks.{n - 1}.hook_resid_post" if layer == n else f"blocks.{layer}.hook_resid_pre"
    ]
    stack = cache.get_full_resid_decomposition(layer=layer, expand_neurons=False)
    torch.testing.assert_close(stack.sum(0), target, atol=ATOL, rtol=0)


def test_use_attn_result_matches_cache_results(cache):
    bridge, ref = _pair()
    bridge.set_use_attn_result(True)
    with torch.no_grad():
        logits, result_cache = bridge.run_with_cache(TOKENS)
        expected = ref(TOKENS).logits
    torch.testing.assert_close(logits, expected, atol=ATOL, rtol=0)
    n_heads = bridge.cfg.n_heads
    for layer in range(bridge.cfg.n_layers):
        from_cache = cache.stack_head_results(layer=layer + 1)[layer * n_heads :]
        torch.testing.assert_close(
            result_cache[f"blocks.{layer}.attn.hook_result"],
            from_cache.permute(1, 2, 0, 3),
            atol=ATOL,
            rtol=0,
        )


def test_greedy_generate_matches_hf(pair):
    bridge, ref = pair
    prompt = TOKENS[:1, :3]
    with torch.no_grad():
        expected = ref.generate(prompt, max_new_tokens=5, do_sample=False)
        out = bridge.generate(
            prompt, max_new_tokens=5, do_sample=False, return_type="tokens", verbose=False
        )
    assert torch.equal(out, expected)
