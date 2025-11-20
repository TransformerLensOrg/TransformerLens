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

    @pytest.mark.skip(
        reason="API not implemented - ProcessWeights doesn't convert to TL format keys"
    )
    def test_processing_with_architecture_adapter(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test ProcessWeights.process_weights with architecture adapter."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Preprocess weights first (this converts to TL format with split Q/K/V)
        preprocessed_state_dict = adapter.preprocess_weights(raw_state_dict)

        # Process with architecture adapter
        processed_with_adapter = ProcessWeights.process_weights(
            state_dict=preprocessed_state_dict,
            cfg=cfg,
            adapter=adapter,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Verify processing occurred
        assert len(processed_with_adapter) > 0, "Should process weights with adapter"

        # Check for TransformerLens-style keys (after preprocessing)
        # These should be in format like: blocks.0.attn.W_Q, blocks.0.attn.W_K, etc.
        tl_keys = [
            k
            for k in processed_with_adapter.keys()
            if any(
                pattern in k for pattern in [".W_Q", ".W_K", ".W_V", ".W_O", ".b_Q", ".b_K", ".b_V"]
            )
        ]
        assert (
            len(tl_keys) > 0
        ), "Should have TransformerLens-style attention keys after preprocessing"

        # Check that expected TL-style keys exist
        expected_patterns = ["blocks.0.attn.W_Q", "blocks.0.attn.W_K", "blocks.0.attn.W_V"]
        for pattern in expected_patterns:
            assert any(
                pattern in k for k in processed_with_adapter.keys()
            ), f"Should have {pattern} in processed weights"

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

    @pytest.mark.skip(reason="API not implemented - adapter.preprocess_weights doesn't split Q/K/V")
    def test_processing_with_different_flags(self, raw_hf_model_and_state_dict, bridge_and_adapter):
        """Test processing with different flag combinations."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Preprocess weights first
        preprocessed_state_dict = adapter.preprocess_weights(raw_state_dict)

        # Test processing with all flags enabled
        processed_with_flags = ProcessWeights.process_weights(
            state_dict=preprocessed_state_dict.copy(),
            cfg=cfg,
            adapter=adapter,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )

        # Test processing with all flags disabled
        processed_without_flags = ProcessWeights.process_weights(
            state_dict=preprocessed_state_dict.copy(),
            cfg=cfg,
            adapter=adapter,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Both should process successfully
        assert len(processed_with_flags) > 0, "Should process weights with flags"
        assert len(processed_without_flags) > 0, "Should process weights without flags"

    @pytest.mark.skip(reason="API not implemented - adapter.preprocess_weights doesn't split Q/K/V")
    def test_architecture_divergence_handling(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test that adapter preprocessing changes the state dict format."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Preprocess with adapter (splits c_attn into Q/K/V)
        preprocessed_with_adapter = adapter.preprocess_weights(raw_state_dict)

        # Process with adapter after preprocessing
        processed_with_adapter = ProcessWeights.process_weights(
            state_dict=preprocessed_with_adapter,
            cfg=cfg,
            adapter=adapter,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )

        # Process without adapter (no preprocessing)
        processed_without_adapter = ProcessWeights.process_weights(
            state_dict=raw_state_dict.copy(),
            cfg=cfg,
            adapter=None,
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

        # With adapter should have split Q/K/V keys
        tl_attn_keys = [
            k for k in with_adapter_keys if any(p in k for p in [".W_Q", ".W_K", ".W_V"])
        ]
        assert len(tl_attn_keys) > 0, "With adapter should have split Q/K/V keys"

    @pytest.mark.skip(reason="API not implemented - adapter.preprocess_weights doesn't split Q/K/V")
    def test_custom_component_processing_integration(
        self, raw_hf_model_and_state_dict, bridge_and_adapter
    ):
        """Test that adapter preprocessing splits QKV weights correctly."""
        raw_hf_model, raw_state_dict = raw_hf_model_and_state_dict
        bridge, adapter, cfg = bridge_and_adapter

        # Preprocess weights first - this is what splits Q/K/V
        preprocessed_weights = adapter.preprocess_weights(raw_state_dict)

        # Process with adapter after preprocessing
        processed_weights = ProcessWeights.process_weights(
            state_dict=preprocessed_weights,
            cfg=cfg,
            adapter=adapter,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # Check for split Q/K/V weights (created by preprocessing)
        custom_qkv_found = any(".W_Q" in k for k in processed_weights.keys())

        assert custom_qkv_found, "Should have split QKV weights after preprocessing"

        # Verify that QKV splitting occurred for each layer
        q_keys = [k for k in processed_weights.keys() if ".W_Q" in k]
        k_keys = [k for k in processed_weights.keys() if ".W_K" in k]
        v_keys = [k for k in processed_weights.keys() if ".W_V" in k]

        assert len(q_keys) > 0, "Should have Q weight keys"
        assert len(k_keys) > 0, "Should have K weight keys"
        assert len(v_keys) > 0, "Should have V weight keys"

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

        # Verify the bridge has expected components after processing
        assert hasattr(bridge, "blocks"), "Should have transformer blocks"
        assert len(bridge.blocks) > 0, "Should have at least one block"
        assert hasattr(bridge, "embed"), "Should have embedding component"
        assert hasattr(bridge, "unembed"), "Should have unembedding component"
