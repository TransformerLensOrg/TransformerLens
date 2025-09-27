#!/usr/bin/env python3
"""
Integration Tests for Default TransformerBridge Folding Behavior
================================================================

This test verifies the key behavior changes made to fix the default folding issue:
1. Default TransformerBridge should NOT apply folding (should be unfolded)
2. enable_compatibility_mode() should apply folding
3. enable_compatibility_mode(no_processing=True) should remain unfolded
4. Behavior should match expected references
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge


class TestDefaultFoldingBehavior:
    """Test class for default TransformerBridge folding behavior."""

    @pytest.fixture(scope="class")
    def model_name(self):
        return "gpt2"

    @pytest.fixture(scope="class")
    def device(self):
        return "cpu"

    @pytest.fixture(scope="class")
    def test_tokens(self):
        return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    @pytest.fixture(scope="class")
    def reference_unfolded(self, model_name, device):
        """Reference unfolded HookedTransformer."""
        return HookedTransformer.from_pretrained_no_processing(model_name, device=device)

    @pytest.fixture(scope="class")
    def reference_folded(self, model_name, device):
        """Reference folded HookedTransformer."""
        return HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )

    def test_default_bridge_is_unfolded(
        self, model_name, device, test_tokens, reference_unfolded, reference_folded
    ):
        """Test that default TransformerBridge is unfolded, not folded."""
        # Create default bridge (should be unfolded)
        bridge_default = TransformerBridge.boot_transformers(model_name, device=device)

        # Check that no processed components exist by default
        assert not hasattr(
            bridge_default, "blocks"
        ), "Default bridge should not have processed components"
        assert not hasattr(
            bridge_default, "_hf_processing_flags"
        ), "Default bridge should not have processing flags"

        # Test computational behavior
        with torch.no_grad():
            default_loss = bridge_default(test_tokens, return_type="loss")
            unfolded_ref_loss = reference_unfolded(test_tokens, return_type="loss")
            folded_ref_loss = reference_folded(test_tokens, return_type="loss")

        # Default should be closer to unfolded than folded
        unfolded_diff = abs(default_loss - unfolded_ref_loss).item()
        folded_diff = abs(default_loss - folded_ref_loss).item()

        assert unfolded_diff < folded_diff, (
            f"Default bridge should be closer to unfolded reference. "
            f"Unfolded diff: {unfolded_diff:.10f}, Folded diff: {folded_diff:.10f}"
        )

        # Unfolded difference should be small (allowing for minor implementation differences)
        assert (
            unfolded_diff < 1e-5
        ), f"Default bridge should closely match unfolded reference (diff: {unfolded_diff:.10f})"

    def test_enable_compatibility_mode_applies_folding(
        self, model_name, device, test_tokens, reference_folded
    ):
        """Test that enable_compatibility_mode() applies folding correctly."""
        # Create bridge and enable compatibility mode
        bridge_folded = TransformerBridge.boot_transformers(model_name, device=device)
        bridge_folded.enable_compatibility_mode()

        # Check that processed components exist after enable_compatibility_mode
        assert hasattr(
            bridge_folded, "blocks"
        ), "Bridge should have processed components after enable_compatibility_mode"
        assert hasattr(
            bridge_folded, "_hf_processing_flags"
        ), "Bridge should have processing flags after enable_compatibility_mode"

        # Check processing flags
        flags = bridge_folded._hf_processing_flags
        assert flags["fold_ln"] is True, "fold_ln should be enabled"
        assert flags["center_writing_weights"] is True, "center_writing_weights should be enabled"
        assert flags["center_unembed"] is True, "center_unembed should be enabled"
        assert flags["fold_value_biases"] is True, "fold_value_biases should be enabled"

        # Test computational behavior
        with torch.no_grad():
            folded_loss = bridge_folded(test_tokens, return_type="loss")
            ref_folded_loss = reference_folded(test_tokens, return_type="loss")

        # Should match folded reference exactly
        folded_diff = abs(folded_loss - ref_folded_loss).item()
        assert (
            folded_diff < 1e-8
        ), f"Folded bridge should exactly match folded reference (diff: {folded_diff:.10f})"

    def test_no_processing_flag_prevents_folding(
        self, model_name, device, test_tokens, reference_unfolded
    ):
        """Test that enable_compatibility_mode(no_processing=True) prevents folding."""
        # Create bridge and enable compatibility mode with no_processing
        bridge_no_proc = TransformerBridge.boot_transformers(model_name, device=device)
        bridge_no_proc.enable_compatibility_mode(no_processing=True)

        # Check that no processed components exist with no_processing=True
        assert not hasattr(
            bridge_no_proc, "blocks"
        ), "Bridge should not have processed components with no_processing=True"

        # Test computational behavior
        with torch.no_grad():
            no_proc_loss = bridge_no_proc(test_tokens, return_type="loss")
            ref_unfolded_loss = reference_unfolded(test_tokens, return_type="loss")

        # Should be close to unfolded reference
        unfolded_diff = abs(no_proc_loss - ref_unfolded_loss).item()
        assert (
            unfolded_diff < 1e-5
        ), f"no_processing bridge should closely match unfolded reference (diff: {unfolded_diff:.10f})"

    def test_behavior_consistency(self, model_name, device, test_tokens):
        """Test that the three configurations have consistent relationships."""
        # Create all three configurations
        bridge_default = TransformerBridge.boot_transformers(model_name, device=device)

        bridge_folded = TransformerBridge.boot_transformers(model_name, device=device)
        bridge_folded.enable_compatibility_mode()

        bridge_no_proc = TransformerBridge.boot_transformers(model_name, device=device)
        bridge_no_proc.enable_compatibility_mode(no_processing=True)

        # Test computational behavior
        with torch.no_grad():
            default_loss = bridge_default(test_tokens, return_type="loss")
            folded_loss = bridge_folded(test_tokens, return_type="loss")
            no_proc_loss = bridge_no_proc(test_tokens, return_type="loss")

        # Check relationships
        default_vs_folded = abs(default_loss - folded_loss).item()
        default_vs_no_proc = abs(default_loss - no_proc_loss).item()
        folded_vs_no_proc = abs(folded_loss - no_proc_loss).item()

        # Default should be identical to no_processing (both unfolded)
        assert (
            default_vs_no_proc < 1e-10
        ), f"Default and no_processing should be identical (diff: {default_vs_no_proc:.10f})"

        # Default should be different from folded (folding changes behavior)
        assert (
            default_vs_folded > 1e-6
        ), f"Default and folded should be different (diff: {default_vs_folded:.10f})"

        # Folded should be different from no_processing (flag prevents folding)
        assert (
            folded_vs_no_proc > 1e-6
        ), f"Folded and no_processing should be different (diff: {folded_vs_no_proc:.10f})"

    def test_folding_impact_magnitude(
        self, model_name, device, test_tokens, reference_unfolded, reference_folded
    ):
        """Test that folding has a measurable but small impact on loss."""
        with torch.no_grad():
            unfolded_loss = reference_unfolded(test_tokens, return_type="loss")
            folded_loss = reference_folded(test_tokens, return_type="loss")

        folding_impact = abs(folded_loss - unfolded_loss).item()

        # Folding should have some impact
        assert (
            folding_impact > 1e-7
        ), f"Folding should have measurable impact (impact: {folding_impact:.10f})"

        # But impact should be small (less than 1%)
        percentage_change = folding_impact / unfolded_loss.item() * 100
        assert (
            percentage_change < 1.0
        ), f"Folding impact should be small (<1%, actual: {percentage_change:.6f}%)"

    def test_normalization_bridge_integration(self, model_name, device):
        """Test that TransformerBridge uses NormalizationBridge instead of LayerNormPre."""
        # Create folded bridge
        bridge_folded = TransformerBridge.boot_transformers(model_name, device=device)
        bridge_folded.enable_compatibility_mode()

        # Import required classes
        from transformer_lens.model_bridge.generalized_components.normalization import (
            NormalizationBridge,
        )

        # Check that ln_final is NormalizationBridge, not LayerNormPre
        assert hasattr(bridge_folded, "ln_final"), "Bridge should have ln_final component"
        assert isinstance(
            bridge_folded.ln_final, NormalizationBridge
        ), f"ln_final should be NormalizationBridge, got {type(bridge_folded.ln_final).__name__}"

        # Verify that the NormalizationBridge has LayerNormPre functionality integrated
        ln_final = bridge_folded.ln_final
        assert hasattr(
            ln_final, "_layernorm_pre_forward"
        ), "NormalizationBridge should have LayerNormPre functionality"
        assert hasattr(
            ln_final.config, "layer_norm_folding"
        ), "NormalizationBridge should have layer_norm_folding config"
        assert ln_final.config.layer_norm_folding, "NormalizationBridge should be in folding mode"

        # Test that the component has hook points like LayerNormPre
        assert hasattr(ln_final, "hook_scale"), "NormalizationBridge should have hook_scale"
        assert hasattr(
            ln_final, "hook_normalized"
        ), "NormalizationBridge should have hook_normalized"

        # Verify computational behavior matches LayerNormPre
        test_input = torch.randn(1, 5, bridge_folded.cfg.d_model)
        with torch.no_grad():
            output = ln_final(test_input)

        # Output should be normalized (mean ~0, std ~1)
        output_mean = output.mean(dim=-1)
        output_std = output.std(dim=-1)

        assert torch.allclose(
            output_mean, torch.zeros_like(output_mean), atol=1e-6
        ), f"Output should be centered, got mean: {output_mean.abs().max().item():.8f}"
        assert torch.allclose(
            output_std, torch.ones_like(output_std), atol=1e-6
        ), f"Output should be normalized, got std deviation from 1: {(output_std - 1).abs().max().item():.8f}"
