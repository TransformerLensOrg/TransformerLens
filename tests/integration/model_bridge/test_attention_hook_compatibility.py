"""Test attention hook behavior between HookedTransformer and TransformerBridge."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge


class TestAttentionHookCompatibility:
    """Test attention hook behavior compatibility."""

    @pytest.fixture(scope="class")
    def models(self):
        """Create HookedTransformer and TransformerBridge for testing."""
        # Create reference model (using distilgpt2 for faster tests)
        reference_model = HookedTransformer.from_pretrained("distilgpt2", device="cpu")

        # Create bridge model
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
        bridge.enable_compatibility_mode()

        return reference_model, bridge

    @pytest.fixture
    def test_input(self, models):
        """Create test input tokens."""
        reference_model, _ = models
        test_text = "The cat sat on"
        return reference_model.to_tokens(test_text)

    def test_hook_shapes_match(self, models, test_input):
        """Test that attention hooks produce matching activation shapes."""
        reference_model, bridge = models
        hook_name = "blocks.0.attn.hook_v"

        # Collect activations from both models
        ref_activations = []
        bridge_activations = []

        def collect_ref_hook(activation, hook):
            ref_activations.append(activation)
            return activation

        def collect_bridge_hook(activation, hook):
            bridge_activations.append(activation)
            return activation

        # Run with hooks
        reference_model.add_hook(hook_name, collect_ref_hook)
        bridge.add_hook(hook_name, collect_bridge_hook)

        with torch.no_grad():
            reference_model(test_input)
            bridge(test_input)

        # Clean up hooks
        reference_model.reset_hooks()
        bridge.reset_hooks()

        # Verify shapes match
        assert len(ref_activations) == 1, "Reference model should have one activation"
        assert len(bridge_activations) == 1, "Bridge should have one activation"
        assert (
            ref_activations[0].shape == bridge_activations[0].shape
        ), f"Activation shapes should match: {ref_activations[0].shape} vs {bridge_activations[0].shape}"

    def test_ablation_hook_works(self, models, test_input):
        """Test that ablation hooks work correctly on both models."""
        reference_model, bridge = models
        hook_name = "blocks.0.attn.hook_v"

        def ablation_hook(activation, hook):
            """Zero out the activation as ablation."""
            return torch.zeros_like(activation)

        # Test reference model ablation
        reference_model.add_hook(hook_name, ablation_hook)
        with torch.no_grad():
            ref_ablated_loss = reference_model(test_input, return_type="loss")
        reference_model.reset_hooks()

        # Test bridge ablation
        bridge.add_hook(hook_name, ablation_hook)
        with torch.no_grad():
            bridge_ablated_loss = bridge(test_input, return_type="loss")
        bridge.reset_hooks()

        # Both ablations should produce reasonable (higher) losses
        assert (
            ref_ablated_loss > 3.0
        ), f"Reference ablated loss should be reasonable: {ref_ablated_loss}"
        assert (
            bridge_ablated_loss > 3.0
        ), f"Bridge ablated loss should be reasonable: {bridge_ablated_loss}"

        # Ablated losses should be close to each other
        diff = abs(ref_ablated_loss - bridge_ablated_loss)
        assert diff < 1.0, f"Ablated losses should match closely: {diff}"

    def test_hook_names_available(self, models):
        """Test that expected hook names are available in both models."""
        reference_model, bridge = models

        expected_hooks = ["blocks.0.attn.hook_v", "blocks.0.attn.hook_q", "blocks.0.attn.hook_k"]

        # Check reference model hooks
        ref_hook_names = set(reference_model.hook_dict.keys())
        for hook_name in expected_hooks:
            assert hook_name in ref_hook_names, f"Reference model missing hook: {hook_name}"

        # Check bridge hooks
        bridge_hook_names = set(bridge.hook_dict.keys())
        for hook_name in expected_hooks:
            assert hook_name in bridge_hook_names, f"Bridge missing hook: {hook_name}"

    def test_hook_error_handling(self, models, test_input):
        """Test that hook errors are handled gracefully."""
        reference_model, bridge = models
        hook_name = "blocks.0.attn.hook_v"

        def error_hook(activation, hook):
            """Hook that raises an error."""
            raise ValueError("Test error in hook")

        # Test error handling in reference model
        reference_model.add_hook(hook_name, error_hook)
        with pytest.raises(ValueError, match="Test error in hook"):
            with torch.no_grad():
                reference_model(test_input)
        reference_model.reset_hooks()

        # Test error handling in bridge
        bridge.add_hook(hook_name, error_hook)
        with pytest.raises(ValueError, match="Test error in hook"):
            with torch.no_grad():
                bridge(test_input)
        bridge.reset_hooks()
