"""Offline regressions for Granite's post-embedding runtime multiplier."""

import copy

import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.tools.analysis import direct_logit_attribution


@pytest.fixture(params=[1.0, 0.5, 2.0, 12.0])
def granite(request):
    torch.manual_seed(0)
    config = GraniteConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        embedding_multiplier=request.param,
        residual_multiplier=0.22,
        logits_scaling=1.0,
        attention_multiplier=0.5,
        attn_implementation="eager",
    )
    model = GraniteForCausalLM(config).eval()
    reference = copy.deepcopy(model)
    bridge = build_bridge_from_module(model, "GraniteForCausalLM", hf_config=config, device="cpu")
    yield bridge, reference, request.param
    bridge.close()


@pytest.mark.parametrize("compatibility", [False, True])
@pytest.mark.parametrize("remove_batch", [False, True])
@pytest.mark.parametrize("expand_neurons", [False, True])
def test_embedding_contribution_reconstructs_residual(
    granite, compatibility, remove_batch, expand_neurons
):
    bridge, _, multiplier = granite
    if compatibility:
        bridge.enable_compatibility_mode()
    assert bridge.cfg.embedding_multiplier == multiplier
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens, remove_batch_dim=remove_batch)
    stack, labels = cache.decompose_resid(return_labels=True)
    torch.testing.assert_close(stack[labels.index("embed")], cache["hook_embed"])
    torch.testing.assert_close(stack.sum(0), cache["blocks.1.hook_out"], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        cache.decompose_resid(layer=0).sum(0),
        cache["blocks.0.hook_in"],
        rtol=1e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        cache.decompose_resid(pos_slice=-1).sum(0),
        cache["blocks.1.hook_out"][..., -1, :],
        rtol=1e-5,
        atol=1e-7,
    )
    full = cache.get_full_resid_decomposition(expand_neurons=expand_neurons)
    torch.testing.assert_close(full.sum(0), cache["blocks.1.hook_out"], rtol=1e-5, atol=1e-7)


def test_component_dla_reconstructs_logit_difference(granite):
    bridge, _, _ = granite
    bridge.enable_compatibility_mode()
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(tokens)
    result = direct_logit_attribution(
        bridge, answer_tokens=7, incorrect_tokens=9, unit="component", cache=cache
    )
    torch.testing.assert_close(
        result.attribution.sum(0),
        logits[:, -1, 7] - logits[:, -1, 9],
        rtol=1e-5,
        atol=1e-7,
    )


def test_embedding_hook_exposes_residual_contribution_without_changing_forward(granite):
    bridge, reference, multiplier = granite
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(tokens)
        torch.testing.assert_close(logits, reference(tokens).logits, rtol=0, atol=0)
        torch.testing.assert_close(
            cache["hook_embed"], reference.model.embed_tokens(tokens) * multiplier
        )
        torch.testing.assert_close(cache["hook_embed"], cache["blocks.0.hook_in"])
        bridge.add_hook("embed.hook_out", lambda value, hook: value * 3)
        try:
            _, edited = bridge.run_with_cache(tokens)
        finally:
            bridge.reset_hooks()
    torch.testing.assert_close(edited["hook_embed"], cache["hook_embed"] * 3)
    torch.testing.assert_close(edited["blocks.0.hook_in"], edited["hook_embed"])


@pytest.mark.parametrize("in_place", [False, True])
def test_embedding_ablation_lands_in_residual_stream(granite, in_place):
    bridge, _, _ = granite

    def ablate(value, hook):
        return value.zero_() if in_place else torch.zeros_like(value)

    bridge.add_hook("embed.hook_out", ablate)
    try:
        with torch.no_grad():
            _, cache = bridge.run_with_cache(torch.tensor([[3, 4, 5]]))
        torch.testing.assert_close(cache["blocks.0.hook_in"], torch.zeros_like(cache["hook_embed"]))
    finally:
        bridge.reset_hooks()


def test_embedding_backward_hook_receives_residual_gradient(granite):
    bridge, _, _ = granite
    gradients = {}

    def record(gradient, hook):
        gradients[hook.name] = gradient.detach().clone()

    bridge.add_hook("embed.hook_out", record, dir="bwd")
    bridge.add_hook("blocks.0.hook_in", record, dir="bwd")
    try:
        bridge(torch.tensor([[3, 4, 5]])).square().sum().backward()
        assert torch.isfinite(gradients["embed.hook_out"]).all()
        assert gradients["embed.hook_out"].abs().sum() > 0
        torch.testing.assert_close(gradients["embed.hook_out"], gradients["blocks.0.hook_in"])
    finally:
        bridge.reset_hooks()


def test_live_and_reconstructed_head_results_have_same_scale(granite):
    bridge, _, _ = granite
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        _, reconstructed = bridge.run_with_cache(tokens)
        reconstructed.compute_head_results()
        bridge.set_use_attn_result(True)
        _, live = bridge.run_with_cache(tokens)
    for layer in range(bridge.cfg.n_layers):
        key = f"blocks.{layer}.attn.hook_result"
        torch.testing.assert_close(live[key], reconstructed[key], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            live[key].sum(-2), live[f"blocks.{layer}.hook_attn_out"], rtol=1e-5, atol=1e-7
        )
    torch.testing.assert_close(
        live.get_full_resid_decomposition(expand_neurons=True).sum(0),
        live["blocks.1.hook_out"],
        rtol=1e-5,
        atol=1e-7,
    )
    bridge.add_hook("blocks.0.attn.hook_result", lambda value, hook: torch.zeros_like(value))
    try:
        with torch.no_grad():
            _, edited = bridge.run_with_cache(tokens)
        torch.testing.assert_close(
            edited["blocks.0.hook_attn_out"], torch.zeros_like(edited["blocks.0.hook_attn_out"])
        )
    finally:
        bridge.reset_hooks()


@pytest.fixture
def biased_granite(granite):
    _, reference, _ = granite
    model = copy.deepcopy(reference)
    # Bind the bias-bearing model before Bridge snapshots its parameter aliases.
    for layer in model.model.layers:
        layer.self_attn.o_proj.bias = torch.nn.Parameter(torch.randn(16) * 0.01)
        layer.mlp.down_proj.bias = torch.nn.Parameter(torch.randn(16) * 0.01)
    bridge = build_bridge_from_module(
        model, "GraniteForCausalLM", hf_config=model.config, device="cpu"
    )
    yield bridge
    bridge.close()


def test_neuron_projection_and_scaled_bias_reconstruct_residual(biased_granite):
    bridge = biased_granite
    with torch.no_grad():
        _, cache = bridge.run_with_cache(torch.tensor([[3, 4, 5]]))
    full = cache.get_full_resid_decomposition(expand_neurons=True)
    torch.testing.assert_close(full.sum(0), cache["blocks.1.hook_out"], rtol=1e-5, atol=1e-7)
    directions = torch.randn(16, 2)
    dense = cache.stack_neuron_results(2, apply_ln=True) @ directions
    projected = cache.stack_neuron_results(2, apply_ln=True, project_output_onto=directions)
    torch.testing.assert_close(projected, dense, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("expand_neurons", [False, True])
def test_mlp_input_decomposition_scales_target_layer_attention_bias(
    biased_granite, layer, expand_neurons
):
    bridge = biased_granite
    with torch.no_grad():
        _, cache = bridge.run_with_cache(torch.tensor([[3, 4, 5]]))
    full = cache.get_full_resid_decomposition(layer, mlp_input=True, expand_neurons=expand_neurons)
    torch.testing.assert_close(
        full.sum(0), cache[f"blocks.{layer}.hook_resid_mid"], rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("multiplier", [0.5, 2.0])
def test_moe_embedding_contribution(multiplier):
    from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

    torch.manual_seed(0)
    config = GraniteMoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_local_experts=2,
        num_experts_per_tok=1,
        embedding_multiplier=multiplier,
        residual_multiplier=0.22,
        logits_scaling=1.0,
        attn_implementation="eager",
    )
    model = GraniteMoeForCausalLM(config).eval()
    bridge = build_bridge_from_module(
        model, "GraniteMoeForCausalLM", hf_config=config, device="cpu"
    )
    try:
        with torch.no_grad():
            _, cache = bridge.run_with_cache(torch.tensor([[3, 4, 5]]))
        torch.testing.assert_close(
            cache.decompose_resid().sum(0),
            cache["blocks.0.hook_out"],
            rtol=1e-5,
            atol=1e-7,
        )
    finally:
        bridge.close()
