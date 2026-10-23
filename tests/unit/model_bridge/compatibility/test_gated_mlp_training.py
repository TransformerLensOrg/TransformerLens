"""Gated MLPs train in compatibility mode like every other component."""

from typing import Any, Dict

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources import build_bridge_from_module

# Phi3 fuses gate/up in HF and the bridge splits them, so it covers JointGateUpMLPBridge.
SPECS: Dict[str, Dict[str, Any]] = {
    "LlamaForCausalLM": dict(model_type="llama"),
    "Phi3ForCausalLM": dict(model_type="phi3", pad_token_id=0),
}


def _compat_bridge(architecture: str) -> TransformerBridge:
    torch.manual_seed(0)
    hf_config = AutoConfig.for_model(
        **SPECS[architecture],
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    hf_config.architectures = [architecture]
    hf_model = AutoModelForCausalLM.from_config(hf_config).to(torch.float32).eval()
    bridge = build_bridge_from_module(hf_model, architecture, hf_config=hf_config)
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge


@pytest.mark.parametrize("architecture", sorted(SPECS))
def test_gated_mlp_projections_receive_gradients(architecture: str) -> None:
    bridge = _compat_bridge(architecture)
    bridge(torch.tensor([[1, 5, 9, 13, 17, 21]])).sum().backward()

    grads = {
        f"blocks.{index}.mlp.{projection}.{name}": param.grad
        for index, block in enumerate(bridge.blocks)
        for projection in ("gate", "in", "out")
        for name, param in getattr(block.mlp, projection).named_parameters()
    }
    assert len(grads) == 3 * len(bridge.blocks)
    missing = [name for name, grad in grads.items() if grad is None or not grad.any()]
    assert not missing, f"no gradient reached {missing}"
