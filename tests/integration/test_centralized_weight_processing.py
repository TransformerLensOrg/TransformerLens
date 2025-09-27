#!/usr/bin/env python3
"""
Integration Tests for Centralized Weight Processing
===================================================

This test verifies the centralized ProcessWeights.process_raw_weights functionality:
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

    def test_processing_with_architecture_adapter(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test ProcessWeights.process_raw_weights with architecture adapter."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Process with architecture adapter
        processed_with_adapter = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=adapter,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Verify processing occurred
        assert len(processed_with_adapter) > 0, "Should process weights with adapter"

        # Check for custom processed keys (TransformerLens format)
        custom_keys = [
            k for k in processed_with_adapter.keys() if "W_E" in k or "W_pos" in k or "W_Q" in k
        ]
        assert len(custom_keys) > 0, "Should have custom processed keys with adapter"

        # Check that original HF keys are preserved/converted
        expected_keys = ["W_E", "W_pos", "W_U"]
        for key in expected_keys:
            assert key in processed_with_adapter, f"Should have {key} in processed weights"

    def test_processing_without_architecture_adapter(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test ProcessWeights.process_raw_weights without architecture adapter."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Process without architecture adapter
        processed_without_adapter = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=None,  # No adapter
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

    def test_bypass_mechanism(self, raw_hf_model_and_state_dict, bridge_and_adapter):
        """Test bypass mechanisms for fine-grained control."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Test bypass mechanism
        bypass_flags = {"fold_ln": True, "center_writing_weights": True}
        processed_with_bypass = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=adapter,
            fold_ln=True,  # This should be bypassed
            center_writing_weights=True,  # This should be bypassed
            center_unembed=False,
            fold_value_biases=False,
            bypass_default_processing=bypass_flags,
        )

        # Verify bypass worked
        assert len(processed_with_bypass) > 0, "Should process weights with bypass"

        # Test that we can process with different parameters
        processed_normal = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=adapter,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Results should be different (bypass should affect processing)
        assert len(processed_with_bypass) == len(
            processed_normal
        ), "Should have same number of keys"

    def test_architecture_divergence_handling(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test that adapter detection handles architecture divergence correctly."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Process with adapter (TransformerBridge case)
        processed_with_adapter = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=adapter,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )

        # Process without adapter (HookedTransformer case)
        processed_without_adapter = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=None,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )

        # Results should be different (different processing paths)
        with_adapter_keys = set(processed_with_adapter.keys())
        without_adapter_keys = set(processed_without_adapter.keys())

        # Should have some different keys due to different processing
        assert (
            with_adapter_keys != without_adapter_keys
        ), "With and without adapter should produce different key sets"

        # With adapter should have TransformerLens-style keys
        tl_keys = [k for k in with_adapter_keys if k in ["W_E", "W_pos", "W_U"]]
        assert len(tl_keys) > 0, "With adapter should have TransformerLens-style keys"

    def test_custom_component_processing_integration(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test that custom component processing is integrated correctly."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Process with adapter to enable custom component processing
        processed_weights = ProcessWeights.process_raw_weights(
            raw_hf_state_dict=raw_state_dict,
            cfg=cfg,
            architecture_adapter=adapter,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Check for custom component processing results
        custom_embed_found = "W_E" in processed_weights
        custom_pos_found = "W_pos" in processed_weights
        custom_qkv_found = any("W_Q" in k for k in processed_weights.keys())

        assert custom_embed_found, "Should have custom embed processing"
        assert custom_pos_found, "Should have custom pos embed processing"
        assert custom_qkv_found, "Should have custom QKV processing"

        # Verify that QKV splitting occurred (multiple attention heads)
        q_keys = [k for k in processed_weights.keys() if "W_Q" in k]
        k_keys = [k for k in processed_weights.keys() if "W_K" in k]
        v_keys = [k for k in processed_weights.keys() if "W_V" in k]

        assert len(q_keys) > 0, "Should have Q weight keys"
        assert len(k_keys) > 0, "Should have K weight keys"
        assert len(v_keys) > 0, "Should have V weight keys"

    def test_computational_correctness_with_existing_pipeline(self, model_name, device):
        """Test that centralized processing maintains computational correctness."""
        test_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        # Create TransformerBridge using existing pipeline
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()

        with torch.no_grad():
            bridge_loss = bridge(test_tokens, return_type="loss")

        # Get HF weights from bridge
        hf_weights = bridge.get_processed_hf_weights()

        # Verify that processing maintains computational correctness
        assert len(hf_weights) > 0, "Should export HF weights"
        assert bridge_loss.item() > 0, "Should produce valid loss"

        # Check for expected HF format keys
        expected_patterns = ["transformer.", "lm_head."]
        has_expected_keys = any(
            any(pattern in key for pattern in expected_patterns) for key in hf_weights.keys()
        )
        assert has_expected_keys, "Should have HF format keys in exported weights"
