#!/usr/bin/env python3
"""Comprehensive test suite for TransformerBridge compatibility with HookedTransformer.

This test suite ensures that TransformerBridge maintains perfect compatibility
with HookedTransformer while using direct weight processing instead of delegation.
"""

import pytest

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks import (
    benchmark_hook_functionality,
    benchmark_hook_registry,
    benchmark_logits_equivalence,
    benchmark_loss_equivalence,
    benchmark_weight_modification,
    benchmark_weight_processing,
    benchmark_weight_sharing,
)
from transformer_lens.model_bridge import TransformerBridge


class TestTransformerBridgeCompatibility:
    """Test TransformerBridge compatibility and behavioral equivalence with HookedTransformer."""

    @pytest.fixture
    def models(self):
        """Create HookedTransformer and TransformerBridge for comparison."""
        device = "cpu"
        model_name = "gpt2"

        # HookedTransformer with processing
        ht = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )

        # TransformerBridge with compatibility mode
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()

        return {"ht": ht, "bridge": bridge}

    @pytest.fixture
    def test_text(self):
        """Test text for forward pass verification."""
        return "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    def test_forward_pass_equivalence(self, models, test_text):
        """Test that forward passes produce identical results."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Use benchmark function
        result = benchmark_loss_equivalence(bridge, test_text, reference_model=ht, atol=1e-3)
        assert result.passed, result.message

    def test_logits_equivalence(self, models, test_text):
        """Test that logits outputs are nearly identical.

        Note: Weights are identical, but forward pass implementations differ slightly,
        leading to accumulated numerical precision differences (~0.02 max).
        """
        ht = models["ht"]
        bridge = models["bridge"]

        # Use benchmark function
        result = benchmark_logits_equivalence(
            bridge, test_text, reference_model=ht, atol=3e-2, rtol=3e-2
        )
        assert result.passed, result.message

    def test_hook_functionality_equivalence(self, models, test_text):
        """Test that hook system produces identical ablation effects."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Use benchmark function
        result = benchmark_hook_functionality(bridge, test_text, reference_model=ht, atol=2e-3)
        assert result.passed, result.message

    def test_weight_sharing_verification(self, models, test_text):
        """Test that modifying weights affects both models similarly."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Use benchmark function
        result = benchmark_weight_sharing(bridge, test_text, reference_model=ht, atol=1e-3)
        assert result.passed, result.message

    def test_component_structure_equivalence(self, models):
        """Test that component structures are equivalent."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Test that core components exist and have correct types
        assert hasattr(bridge, "embed"), "Bridge should have embed component"
        assert hasattr(bridge, "blocks"), "Bridge should have blocks component"
        assert hasattr(bridge, "ln_final"), "Bridge should have ln_final component"
        assert hasattr(bridge, "unembed"), "Bridge should have unembed component"

        # Test component structure matches
        assert len(bridge.blocks) == len(ht.blocks), "Should have same number of blocks"

        # Test weight shapes match
        assert (
            ht.embed.W_E.shape == bridge.embed.W_E.shape
        ), "Embedding weights should have same shape"
        assert (
            ht.blocks[0].attn.W_V.shape == bridge.blocks[0].attn.W_V.shape
        ), "Attention weights should have same shape"

    def test_weight_processing_verification(self, models):
        """Test that weight processing (folding, centering) was applied correctly."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Use benchmark function
        result = benchmark_weight_processing(bridge, test_text="", reference_model=ht)
        assert result.passed, result.message

    def test_hook_registry_completeness(self, models):
        """Test that TransformerBridge has complete hook registry from HookedTransformer."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Use benchmark function
        result = benchmark_hook_registry(bridge, reference_model=ht)
        assert result.passed, result.message

    def test_no_persistent_hookedtransformer_reference(self, models):
        """Test that TransformerBridge has no persistent HookedTransformer references."""
        bridge = models["bridge"]

        # Should not have any persistent HookedTransformer references
        assert not hasattr(
            bridge, "_processed_hooked_transformer"
        ), "Should not have persistent HookedTransformer reference"

        # Should have extracted components
        assert hasattr(bridge, "blocks"), "Should have extracted blocks"
        assert hasattr(bridge, "_hook_registry"), "Should have hook registry"
        assert len(bridge._hook_registry) > 0, "Hook registry should not be empty"


class TestTransformerBridgeWeightModification:
    """Test that TransformerBridge properly responds to weight modifications."""

    @pytest.fixture
    def bridge_model(self):
        """Create TransformerBridge for testing."""
        device = "cpu"
        model_name = "gpt2"

        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()
        return bridge

    def test_weight_modification_propagates(self, bridge_model):
        """Test that weight modifications affect forward pass."""
        test_text = "Natural language processing"

        # Use benchmark function
        result = benchmark_weight_modification(bridge_model, test_text)
        assert result.passed, result.message


# Test cases that can be run individually for debugging
def test_simple_forward_equivalence():
    """Simple standalone test for forward pass equivalence."""
    device = "cpu"
    model_name = "gpt2"
    test_text = "The quick brown fox"

    # Create models
    ht = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode()

    # Test forward pass
    ht_loss = ht(test_text, return_type="loss")
    bridge_loss = bridge(test_text, return_type="loss")

    print(f"HT loss: {ht_loss:.6f}")
    print(f"Bridge loss: {bridge_loss:.6f}")
    print(f"Difference: {abs(ht_loss - bridge_loss):.6f}")

    assert abs(ht_loss - bridge_loss) < 2e-3


if __name__ == "__main__":
    # Run simple test when executed directly
    test_simple_forward_equivalence()
    print("✅ Simple forward equivalence test passed!")
