"""Native normalization interventions through a tiny real Llama forward, without downloads."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(scope="module")
def tiny_llama():
    with torch.random.fork_rng():
        torch.manual_seed(3)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        config._attn_implementation = "eager"
        hf = LlamaForCausalLM(config).eval()
        tokens = torch.tensor([[2, 3, 4]])
        with torch.no_grad():
            reference = hf(tokens).logits
        bridge = build_bridge_from_module(hf, "LlamaForCausalLM", hf_config=config, device="cpu")
    return bridge, tokens, reference


@pytest.mark.parametrize(
    "mode", [torch.no_grad, torch.inference_mode], ids=["no-grad", "inference"]
)
@pytest.mark.parametrize(
    "name",
    [
        "ln_final.hook_normalized",
        "blocks.0.ln1.hook_normalized",
        "ln_final.hook_scale",
        "blocks.0.ln1.hook_scale",
    ],
)
def test_llama_inplace_norm_intervention_matches_replacement(tiny_llama, name, mode):
    bridge, tokens, reference = tiny_llama
    with mode():
        assert torch.equal(bridge(tokens), reference)
        observed, _ = bridge.run_with_cache(tokens)
        assert torch.equal(observed, reference)
        with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
            expected = bridge.run_with_hooks(
                tokens, fwd_hooks=[(name, lambda value, hook: value * 2)]
            )
        with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
            actual = bridge.run_with_hooks(
                tokens, fwd_hooks=[(name, lambda value, hook: value.mul_(2))]
            )
        torch.testing.assert_close(actual, expected)
        assert not torch.equal(actual, reference)
        assert torch.equal(bridge(tokens), reference)
