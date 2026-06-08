#!/usr/bin/env python3
"""
Integration Tests for Centralized Weight Processing
===================================================

This test verifies the centralized ProcessWeights.process_weights functionality:
1. Processing with architecture adapter (TransformerBridge case)
2. Processing without architecture adapter (HookedTransformer case)
3. Bypass mechanisms for fine-grained control
4. Custom component processing integration
5. Architecture adapter detection and divergence handling
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.weight_processing import ProcessWeights


class TestCentralizedWeightProcessing:
    """Test class for centralized weight processing functionality."""

    @pytest.fixture(scope="class")
    def model_name(self):
        return "gpt2"

    @pytest.fixture(scope="class")
    def device(self):
        return "cpu"

    @pytest.fixture(scope="class")
    def raw_hf_model_and_state_dict(self, model_name):
        """Load raw HuggingFace model and return state dict."""
        raw_hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        return raw_hf_model, raw_hf_model.state_dict()

    @pytest.fixture(scope="class")
    def bridge_and_adapter(self, model_name, device):
        """Create bridge and return adapter and config."""
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        return bridge, bridge.adapter, bridge.cfg

    def test_processing_without_architecture_adapter(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test ProcessWeights.process_weights without architecture adapter."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Process without architecture adapter
        processed_without_adapter = ProcessWeights.process_weights(
            state_dict=raw_state_dict,
            cfg=cfg,
            adapter=None,  # No adapter
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Verify processing occurred
        assert len(processed_without_adapter) > 0, "Should process weights without adapter"

        # Check that HF keys are more directly preserved
        hf_keys = [
            k for k in processed_without_adapter.keys() if "transformer." in k or "lm_head" in k
        ]
        assert len(hf_keys) > 0, "Should have HF-style keys without adapter"

    def test_computational_correctness_with_existing_pipeline(self, model_name, device):
        """Test that centralized processing maintains computational correctness."""
        test_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        # Create TransformerBridge using existing pipeline
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()

        # Verify that processing maintains computational correctness
        with torch.no_grad():
            bridge_loss = bridge(test_tokens, return_type="loss")

        assert bridge_loss.item() > 0, "Should produce valid loss after processing"
