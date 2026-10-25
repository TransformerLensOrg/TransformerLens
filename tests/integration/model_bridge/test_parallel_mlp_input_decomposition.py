"""MLP-input decomposition against real tiny NeoX normalization inputs, without Hub loads."""

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(scope="module", params=[False, True], ids=["sequential", "parallel"])
def model_and_cache(request):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        config = GPTNeoXConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=16,
            rotary_pct=0.5,
            use_parallel_residual=request.param,
        )
        config._attn_implementation = "eager"
        hf = GPTNeoXForCausalLM(config).eval()
        with torch.no_grad():
            for block in hf.gpt_neox.layers:
                block.attention.dense.bias.copy_(torch.linspace(-0.4, 0.4, 16))
                block.mlp.dense_4h_to_h.bias.copy_(torch.linspace(0.1, 0.2, 16))
        bridge = build_bridge_from_module(
            hf, "GPTNeoXForCausalLM", hf_config=config, dtype=torch.float32, device="cpu"
        )
        with torch.no_grad():
            _, cache = bridge.run_with_cache(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    return bridge, cache, request.param


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("batchless", [False, True])
def test_component_and_full_decomposition_match_actual_mlp_input(
    model_and_cache, layer, normalized, batchless
):
    bridge, cache, parallel = model_and_cache
    if batchless:
        cache = cache.apply_slice_to_batch_dim(0)
    target = cache[f"blocks.{layer}.ln2.hook_{'out' if normalized else 'in'}"]
    components, labels = cache.decompose_resid(
        layer=layer, mlp_input=True, apply_ln=normalized, return_labels=True
    )
    torch.testing.assert_close(components.sum(0), target)
    assert (f"{layer}_attn_out" in labels) is not parallel
    for expand in (False, True):
        full = cache.get_full_resid_decomposition(
            layer=layer, mlp_input=True, expand_neurons=expand, apply_ln=normalized
        )
        torch.testing.assert_close(full.sum(0), target, atol=2e-5, rtol=1e-5)
        directions = torch.arange(32, dtype=torch.float32).reshape(16, 2) / 32
        projected = cache.get_full_resid_decomposition(
            layer=layer,
            mlp_input=True,
            expand_neurons=expand,
            apply_ln=normalized,
            project_output_onto=directions,
        )
        torch.testing.assert_close(projected.sum(0), target @ directions, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("include_mlp", [False, True])
def test_accumulated_bias_stops_before_parallel_attention(model_and_cache, layer, include_mlp):
    bridge, _, parallel = model_and_cache
    expected = torch.zeros(16)
    for block in bridge.blocks[:layer]:
        expected += block.attn.b_O
        if include_mlp:
            expected += block.mlp.b_out
    if not parallel:
        expected += bridge.blocks[layer].attn.b_O
    torch.testing.assert_close(
        bridge.accumulated_bias(layer, mlp_input=True, include_mlp_biases=include_mlp), expected
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_accumulated_resid_uses_existing_streams(model_and_cache, layer):
    _, cache, parallel = model_and_cache
    parts, labels = cache.accumulated_resid(
        layer=layer, mlp_input=True, incl_mid=True, return_labels=True
    )
    torch.testing.assert_close(parts[-1], cache[f"blocks.{layer}.ln2.hook_in"])
    assert any(label.endswith("_mid") for label in labels) is not parallel


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("projected", ["none", "vector", "matrix"])
@pytest.mark.parametrize("batchless", [False, True])
def test_neuron_remainder_matches_mlp_input(
    model_and_cache, layer, normalized, projected, batchless
):
    _, cache, _ = model_and_cache
    if batchless:
        cache = cache.apply_slice_to_batch_dim(0)
    direction = None
    if projected == "vector":
        direction = torch.arange(16, dtype=torch.float32) / 16
    elif projected == "matrix":
        direction = torch.arange(32, dtype=torch.float32).reshape(16, 2) / 32
    parts = cache.stack_neuron_results(
        layer=layer,
        mlp_input=True,
        incl_remainder=True,
        apply_ln=normalized,
        pos_slice=-1,
        project_output_onto=direction,
    )
    target = cache[f"blocks.{layer}.ln2.hook_{'out' if normalized else 'in'}"][..., -1, :]
    if direction is not None:
        target = target @ direction
    torch.testing.assert_close(parts.sum(0), target, atol=2e-5, rtol=1e-5)
