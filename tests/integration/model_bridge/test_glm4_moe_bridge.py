"""Integration tests for the GLM-4.5 MoE TransformerBridge."""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import MoEBridge

MODEL_ID = "trl-internal-testing/tiny-Glm4MoeForCausalLM"


@pytest.fixture(scope="module")
def tiny_glm4_moe_bridge():
    """Load tiny GLM-4 MoE model via Hub."""

    return TransformerBridge.boot_transformers(
        MODEL_ID,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def tiny_glm4_moe_hf() -> Any:
    """Load the raw HF model for parity checks."""
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()

    # Match the bridge's eager attention path exactly.
    if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
        hf_model.config._attn_implementation = "eager"
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        for layer in hf_model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                layer.self_attn.config._attn_implementation = "eager"

    return hf_model


@pytest.fixture(scope="module")
def tiny_tokens():
    return torch.tensor([[1, 2, 3, 4]])


class TestGlm4MoeBridgeStructure:
    def test_blocks_have_moe_mlp(self, tiny_glm4_moe_bridge) -> None:
        assert len(tiny_glm4_moe_bridge.blocks) > 0
        for i, block in enumerate(tiny_glm4_moe_bridge.blocks):
            assert isinstance(block.mlp, MoEBridge), f"blocks.{i}.mlp is not MoEBridge"

    def test_required_top_level_fields(self, tiny_glm4_moe_bridge) -> None:
        assert hasattr(tiny_glm4_moe_bridge, "embed")
        assert hasattr(tiny_glm4_moe_bridge, "ln_final")
        assert hasattr(tiny_glm4_moe_bridge, "unembed")
        assert tiny_glm4_moe_bridge.cfg.final_rms is True
        assert tiny_glm4_moe_bridge.cfg.normalization_type == "RMS"

    def test_block_attn_has_q_norm_and_k_norm(self, tiny_glm4_moe_bridge) -> None:
        block = tiny_glm4_moe_bridge.blocks[0]
        assert hasattr(block.attn, "q_norm")
        assert hasattr(block.attn, "k_norm")


def test_forward_matches_hf(tiny_glm4_moe_bridge, tiny_glm4_moe_hf: Any, tiny_tokens) -> None:
    """Bridge logits should match HuggingFace on the tiny checkpoint."""
    with torch.no_grad():
        bridge_out = tiny_glm4_moe_bridge(tiny_tokens)
        hf_out = tiny_glm4_moe_hf(tiny_tokens).logits.float()
    max_diff = (bridge_out - hf_out).abs().max().item()
    assert max_diff < 1e-4
