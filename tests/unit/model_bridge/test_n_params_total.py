"""Tests for ``TransformerBridge.n_params_total`` on real model layouts."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge


@pytest.mark.parametrize("model_name", ["gpt2", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"])
def test_n_params_total_matches_uninstrumented_model(model_name: str) -> None:
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, attn_implementation="eager"
    )
    expected = sum(parameter.numel() for parameter in hf_model.parameters())
    bridge = TransformerBridge.boot_transformers(model_name, hf_model=hf_model)
    bridge.enable_compatibility_mode()
    assert bridge.n_params_total == expected

    if "Qwen2" in model_name:
        tl_parameters = bridge.tl_parameters()
        for name in ("W_K", "W_V"):
            weight = tl_parameters[f"blocks.0.attn.{name}"]
            assert weight.shape[0] == bridge.cfg.n_heads
            assert torch.count_nonzero(weight)
        assert not torch.count_nonzero(tl_parameters["pos_embed.W_pos"])
        assert bridge.n_params_total < sum(p.numel() for p in tl_parameters.values())
