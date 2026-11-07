"""A caller's 4D attention mask is the complete mask, as HF treats it.

A [batch, 1, 1, seq] mask is ambiguous: GPT-1's HF module passes that shape as a
padding-only mask, but from a caller it is the whole mask, broadcast over the
query rows. Re-imposing the local causal mask on it made the bridge diverge from
the model it wraps.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from transformer_lens.model_bridge.sources import build_bridge_from_module

SEQ = 5


def _gpt2():
    config = GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=16)
    config.architectures = ["GPT2LMHeadModel"]
    return GPT2LMHeadModel(config), config, "GPT2LMHeadModel"


def _llama():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    config.architectures = ["LlamaForCausalLM"]
    return LlamaForCausalLM(config), config, "LlamaForCausalLM"


@pytest.fixture(scope="module", params=[_gpt2, _llama], ids=["gpt2", "llama"])
def models(request):
    """A tiny bridge and an unwrapped copy: the bridge wraps the blocks it is given."""
    torch.manual_seed(0)
    hf_model, config, architecture = request.param()
    hf_model.eval()
    reference = copy.deepcopy(hf_model)
    return build_bridge_from_module(hf_model, architecture, hf_config=config), reference


def _masks() -> dict[str, torch.Tensor]:
    hole = torch.ones(1, 1, 1, SEQ, dtype=torch.bool)
    hole[..., 1] = False
    return {
        "bool_all_visible": torch.ones(1, 1, 1, SEQ, dtype=torch.bool),
        "additive_all_visible": torch.zeros(1, 1, 1, SEQ),
        "bool_key_hole": hole,
    }


@pytest.mark.parametrize("mask_name", list(_masks()))
def test_broadcast_4d_mask_matches_hf(models, mask_name) -> None:
    bridge, reference = models
    mask = _masks()[mask_name]
    tokens = torch.tensor([[11, 12, 13, 14, 15]])
    with torch.no_grad():
        logits = bridge(tokens, attention_mask=mask, prepend_bos=False)
        expected = reference(tokens, attention_mask=mask).logits
        causal = reference(tokens).logits
    # The mask must actually lift causality on the HF side, or parity proves nothing.
    assert not torch.allclose(expected, causal, atol=1e-3)
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
