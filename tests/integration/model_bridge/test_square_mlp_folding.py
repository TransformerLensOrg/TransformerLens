"""Compatibility-mode folding preserves predictions when the MLP is square (d_mlp == d_model).

A square MLP weight's shape no longer tells its layout apart: GPT-2's Conv1D stores
[in, out], nn.Linear stores [out, in], and adapters with an explicit conversion (Gemma 2)
hand weight processing TL's [d_model, d_mlp]. Folding ln2 into W_in and centering W_in and
W_out must still act on the d_model axis of each.
"""

import copy

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    GPT2Config,
    GPTNeoXConfig,
    LlamaConfig,
)

from transformer_lens.model_bridge.sources import build_bridge_from_module

_LLAMA_LIKE = dict(
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=32,
    vocab_size=32,
)

CASES = [
    pytest.param(
        "GPT2LMHeadModel",
        GPT2Config,
        lambda d_mlp: dict(
            n_embd=16, n_layer=2, n_head=4, n_inner=d_mlp, n_positions=32, vocab_size=32
        ),
        id="gpt2",
    ),
    pytest.param(
        "GPTNeoXForCausalLM",
        GPTNeoXConfig,
        lambda d_mlp: dict(_LLAMA_LIKE, intermediate_size=d_mlp),
        id="gpt_neox",
    ),
    pytest.param(
        "Gemma2ForCausalLM",
        Gemma2Config,
        lambda d_mlp: dict(_LLAMA_LIKE, intermediate_size=d_mlp, head_dim=4, num_key_value_heads=4),
        id="gemma2",
    ),
    pytest.param(
        "LlamaForCausalLM",
        LlamaConfig,
        lambda d_mlp: dict(_LLAMA_LIKE, intermediate_size=d_mlp, num_key_value_heads=4),
        id="llama",
    ),
]


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", CASES)
@pytest.mark.parametrize("d_mlp", [16, 32])
@pytest.mark.parametrize("center_writing_weights", [False, True])
def test_compatibility_mode_matches_hf_with_square_mlp(
    hf_architecture, config_cls, config_kwargs, d_mlp, center_writing_weights
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = config_cls(**config_kwargs(d_mlp))
        config._attn_implementation = "eager"
        hf = AutoModelForCausalLM.from_config(config).eval()
        with torch.no_grad():
            for name, param in hf.named_parameters():
                if "norm" in name or "ln" in name:
                    param.add_(torch.randn_like(param) * 0.3)
        tokens = torch.randint(0, 32, (2, 6))
        with torch.no_grad():
            expected = hf(tokens).logits.log_softmax(-1)
        bridge = build_bridge_from_module(
            copy.deepcopy(hf), hf_architecture, hf_config=copy.deepcopy(config), device="cpu"
        ).eval()
        bridge.enable_compatibility_mode(
            fold_ln=True,
            center_writing_weights=center_writing_weights,
            center_unembed=False,
            fold_value_biases=False,
        )
        with torch.no_grad():
            actual = bridge(tokens).log_softmax(-1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
