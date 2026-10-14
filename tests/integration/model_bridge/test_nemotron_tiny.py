"""Offline Nemotron LayerNorm1P hook regressions using a tiny real HF model."""

import pytest
import torch
from transformers import NemotronConfig, NemotronForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture
def tiny_nemotron():
    config = NemotronConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        norm_eps=1e-5,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        model = NemotronForCausalLM(config).eval()
    tokens = torch.tensor([[1, 7, 8, 9]])
    with torch.no_grad():
        reference = model(tokens).logits
    bridge = build_bridge_from_module(model, architecture="NemotronForCausalLM", hf_config=config)
    return bridge, tokens, reference


@pytest.mark.parametrize("norm_name", ["blocks.0.ln1", "blocks.0.ln2", "ln_final"])
@pytest.mark.parametrize("hook_name", ["hook_scale", "hook_normalized"])
@pytest.mark.parametrize("direction", ["fwd", "bwd"])
def test_layernorm1p_identity_hooks_preserve_logits(tiny_nemotron, norm_name, hook_name, direction):
    bridge, tokens, reference = tiny_nemotron
    norm = bridge.get_submodule(norm_name)
    assert not norm.uses_rms_norm
    assert norm.uses_weight_offset
    fired = []

    def identity(value, hook):
        fired.append(True)
        return value.clone()

    key = f"{norm_name}.{hook_name}"
    message = "Backward hooks" if direction == "bwd" else "reconstructed from the hooked values"
    with bridge.hooks(**{f"{direction}_hooks": [(key, identity)]}):
        with pytest.warns(UserWarning, match=message):
            actual = bridge(tokens)
        if direction == "bwd":
            actual.square().sum().backward()
    assert fired
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
