"""Integration tests for Cohere2 architecture adapter (Cohere2ForCausalLM).

Model: trl-internal-testing/tiny-Cohere2ForCausalLM
  - 2 layers, CPU-safe, no gated access.
  - Current config has only sliding_attention layers, so full-attention NoPE is
    covered by the synthetic unit tests in test_cohere2_adapter.py.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.cohere import (
    _Cohere2AttentionBridge,
)

pytestmark = pytest.mark.slow

MODEL = "trl-internal-testing/tiny-Cohere2ForCausalLM"


@pytest.fixture(scope="module")
def cohere2_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def cohere2_hf() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


class TestCohere2BridgeCreation:
    def test_boot_transformers_succeeds(self, cohere2_bridge: TransformerBridge) -> None:
        assert cohere2_bridge is not None

    def test_block_count_matches_hf(
        self, cohere2_bridge: TransformerBridge, cohere2_hf: torch.nn.Module
    ) -> None:
        assert len(cohere2_bridge.blocks) == cohere2_hf.config.num_hidden_layers

    def test_cfg_layer_types_match_hf(
        self, cohere2_bridge: TransformerBridge, cohere2_hf: torch.nn.Module
    ) -> None:
        assert cohere2_bridge.cfg.layer_types == cohere2_hf.config.layer_types

    def test_attention_bridge_is_cohere2_nope_aware(
        self, cohere2_bridge: TransformerBridge
    ) -> None:
        for block in cohere2_bridge.blocks:
            assert type(block.attn) is _Cohere2AttentionBridge


class TestCohere2MatchesHuggingFace:
    def test_forward_logits_match_hf(
        self, cohere2_bridge: TransformerBridge, cohere2_hf: torch.nn.Module
    ) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_logits = cohere2_bridge(tokens)
            hf_logits = cohere2_hf(tokens).logits
        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < 1e-5, f"Cohere2 bridge vs HF max diff = {max_diff:.6f}"
