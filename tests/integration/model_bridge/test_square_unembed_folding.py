"""Final norm folding preserves predictions even when the readout is square."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.mark.parametrize("vocab_size", [16, 17])
@pytest.mark.parametrize("affine", ["gamma", "beta", "both", "identity"])
def test_final_ln_folding_preserves_logits(vocab_size, affine):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=16,
            n_layer=1,
            n_head=4,
            n_positions=16,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
        config._attn_implementation = "eager"
        hf = GPT2LMHeadModel(config).eval()
        with torch.no_grad():
            if affine in ("gamma", "both"):
                hf.transformer.ln_f.weight.copy_(torch.linspace(0.3, 2.0, 16))
            if affine in ("beta", "both"):
                hf.transformer.ln_f.bias.copy_(torch.linspace(-0.2, 0.3, 16))
        bridge = build_bridge_from_module(
            hf, "GPT2LMHeadModel", hf_config=config, device="cpu"
        ).eval()
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        with torch.no_grad():
            before = bridge(tokens)
            weight = bridge.unembed.W_U.clone()
            bias = bridge.unembed.b_U.clone()
            gamma = hf.transformer.ln_f.weight.clone()
            beta = hf.transformer.ln_f.bias.clone()
            bridge.enable_compatibility_mode(
                fold_ln=True,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
            )
            after = bridge(tokens)
    expected_weight = gamma[:, None] * weight
    expected_weight -= expected_weight.mean(0, keepdim=True)
    torch.testing.assert_close(bridge.unembed.W_U, expected_weight, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(bridge.unembed.b_U, bias + beta @ weight, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(after.log_softmax(-1), before.log_softmax(-1), atol=1e-6, rtol=1e-5)
