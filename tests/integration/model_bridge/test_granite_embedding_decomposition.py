"""Offline regressions for Granite's post-embedding runtime multiplier."""

import copy

import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.tools.analysis import direct_logit_attribution


@pytest.fixture(params=[1.0, 0.5, 2.0])
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
        residual_multiplier=1.0,
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
def test_embedding_contribution_reconstructs_residual(granite, compatibility, remove_batch):
    bridge, _, multiplier = granite
    if compatibility:
        bridge.enable_compatibility_mode()
    assert bridge.cfg.embedding_multiplier == multiplier
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens, remove_batch_dim=remove_batch)
    stack, labels = cache.decompose_resid(return_labels=True)
    torch.testing.assert_close(stack[labels.index("embed")], cache["hook_embed"] * multiplier)
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
    full = cache.get_full_resid_decomposition(expand_neurons=False)
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


def test_native_embedding_hook_and_forward_are_unchanged(granite):
    bridge, reference, multiplier = granite
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(tokens)
        torch.testing.assert_close(logits, reference(tokens).logits, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(cache["hook_embed"], reference.model.embed_tokens(tokens))
        bridge.add_hook("embed.hook_out", lambda value, hook: value * 3)
        try:
            _, edited = bridge.run_with_cache(tokens)
        finally:
            bridge.reset_hooks()
    torch.testing.assert_close(edited["hook_embed"], cache["hook_embed"] * 3)
    torch.testing.assert_close(edited["blocks.0.hook_in"], edited["hook_embed"] * multiplier)


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
        residual_multiplier=1.0,
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
