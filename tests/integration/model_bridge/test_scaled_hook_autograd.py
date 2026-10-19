"""Scaled interventions on real config-built Granite models, without Hub access."""

import copy

import pytest
import torch

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("site", ["embedding", "attention", "mlp", "head_result"])
def test_zero_delta_matches_unscaled_hf_intervention(site, frozen):
    from transformers import GraniteConfig, GraniteForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(4)
        config = GraniteConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            embedding_multiplier=2.0,
            residual_multiplier=0.22,
            logits_scaling=1.0,
            attention_multiplier=0.25,
        )
        config._attn_implementation = "eager"
        model = GraniteForCausalLM(config).eval()
    reference = copy.deepcopy(model)
    model.requires_grad_(not frozen)
    reference.requires_grad_(not frozen)
    bridge = build_bridge_from_module(
        model, "GraniteForCausalLM", hf_config=config, dtype=torch.float32, device="cpu"
    )
    names = {
        "embedding": "hook_embed",
        "attention": "blocks.0.hook_attn_out",
        "mlp": "blocks.0.hook_mlp_out",
        "head_result": "blocks.0.attn.hook_result",
    }
    if site == "head_result":
        bridge.set_use_attn_result(True)
    tokens = torch.tensor([[2, 3, 4]])
    shape = (1, 3, 2, 16) if site == "head_result" else (1, 3, 16)
    delta = torch.nn.Parameter(torch.zeros(shape))
    reference_delta = torch.nn.Parameter(torch.zeros(shape))
    scale = 2.0 if site == "embedding" else 0.22
    if site == "embedding":
        module = reference.model.embed_tokens
    elif site == "mlp":
        module = reference.model.layers[0].mlp
    else:
        module = reference.model.layers[0].self_attn

    def reference_edit(module, args, output):
        edit = reference_delta.sum(dim=-2) if site == "head_result" else reference_delta
        if isinstance(output, tuple):
            return (output[0] + edit / scale, *output[1:])
        return output + edit / scale

    handle = module.register_forward_hook(reference_edit)
    try:
        expected = reference(tokens, use_cache=False).logits
    finally:
        handle.remove()
    actual = bridge.run_with_hooks(
        tokens, fwd_hooks=[(names[site], lambda value, hook: value + delta)]
    )
    torch.testing.assert_close(actual, expected)
    actual[0, -1, 7].backward()
    expected[0, -1, 7].backward()
    assert delta.grad is not None
    assert reference_delta.grad is not None
    assert delta.grad.norm() > 0
    torch.testing.assert_close(delta.grad, reference_delta.grad)
