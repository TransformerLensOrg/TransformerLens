"""Test that TransformerBridge matches HuggingFace model outputs.

These tests verify numerical equivalence between Bridge and the raw HF model
at the component level (MLP, attention) and full-model level (logits).
"""

import pytest
import torch
from transformers import AutoModelForCausalLM


@pytest.fixture()
def bridge_model(gpt2_bridge):
    """Alias session fixture for backward compatibility with test signatures."""
    return gpt2_bridge


@pytest.fixture(scope="module")
def hf_model():
    return AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")


def test_mlp_outputs_match(bridge_model, hf_model):
    """Bridge MLP must produce identical outputs to HF MLP for each layer."""
    test_tensor = torch.randn(3, 5, bridge_model.cfg.d_model)

    for layer_n in range(bridge_model.cfg.n_layers):
        bridge_out = bridge_model.blocks[layer_n].mlp(test_tensor)
        hf_out = hf_model.transformer.h[layer_n].mlp(test_tensor)

        assert torch.allclose(
            bridge_out, hf_out, atol=1e-4
        ), f"MLP layer {layer_n}: max_diff={torch.max(torch.abs(bridge_out - hf_out)):.6f}"


def test_attention_outputs_match(bridge_model, hf_model):
    """Bridge attention must produce identical outputs to HF attention for each layer."""
    input_tensor = torch.randn(3, 5, bridge_model.cfg.d_model)

    for layer_n in range(bridge_model.cfg.n_layers):
        bridge_out = bridge_model.blocks[layer_n].attn(hidden_states=input_tensor)
        hf_out = hf_model.transformer.h[layer_n].attn(hidden_states=input_tensor)

        # Both may return tuples — extract the hidden states
        bridge_tensor = bridge_out[0] if isinstance(bridge_out, tuple) else bridge_out
        hf_tensor = hf_out[0] if isinstance(hf_out, tuple) else hf_out

        assert torch.allclose(
            bridge_tensor, hf_tensor, atol=1e-4
        ), f"Attention layer {layer_n}: max_diff={torch.max(torch.abs(bridge_tensor - hf_tensor)):.6f}"


def test_full_model_logits_match(bridge_model, hf_model):
    """Bridge full-model logits must match HF logits within tight tolerance."""
    tokens = bridge_model.to_tokens("The capital of France is")

    with torch.no_grad():
        bridge_logits = bridge_model(tokens)
        hf_logits = hf_model(tokens).logits

    assert (
        bridge_logits.shape == hf_logits.shape
    ), f"Shape mismatch: {bridge_logits.shape} vs {hf_logits.shape}"
    # Use 1e-4 tolerance — if this needs to be looser, there's a real problem
    assert torch.allclose(
        bridge_logits, hf_logits, atol=1e-4
    ), f"Logits max_diff={torch.max(torch.abs(bridge_logits - hf_logits)):.6f}"


def test_config_dimensions_match(bridge_model, hf_model):
    """Bridge config dimensions must exactly match HF config."""
    assert bridge_model.cfg.n_layers == hf_model.config.n_layer
    assert bridge_model.cfg.d_model == hf_model.config.n_embd
    assert bridge_model.cfg.n_heads == hf_model.config.n_head
    assert bridge_model.cfg.d_vocab == hf_model.config.vocab_size


def test_weight_properties_have_correct_shapes(bridge_model):
    """Weight stacking properties must have shapes consistent with config."""
    cfg = bridge_model.cfg

    assert bridge_model.W_Q.shape[0] == cfg.n_layers
    assert bridge_model.W_K.shape[0] == cfg.n_layers
    assert bridge_model.W_V.shape[0] == cfg.n_layers
    assert bridge_model.W_O.shape[0] == cfg.n_layers
    assert bridge_model.W_in.shape[0] == cfg.n_layers
    assert bridge_model.W_out.shape[0] == cfg.n_layers

    # Weights must be non-zero (not default-initialized garbage)
    assert bridge_model.W_Q.abs().sum() > 0
    assert bridge_model.W_in.abs().sum() > 0
